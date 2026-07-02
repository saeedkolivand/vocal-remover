use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

// Holds the Python (Flask) child process so we can kill it when the app exits.
struct Backend(Mutex<Option<Child>>);

// Shown in the splash (dist/index.html) if the backend never comes up, so the user
// isn't stranded on a blank/dead page. dist/index.html has one <h1> and one <p>.
const BOOT_FAILED_JS: &str = "var h=document.querySelector('h1'),p=document.querySelector('p');\
if(h)h.textContent='Couldn\\u2019t start the engine';\
if(p)p.textContent='The audio backend didn\\u2019t respond. See vocal-remover-backend.log in your system temp folder, then reopen the app.';";

/// Grab a free localhost port from the OS, then release it for the backend to bind.
/// ponytail: tiny TOCTOU window between drop and the child binding — acceptable for a
/// local single-user app; a pipe handshake isn't worth the complexity.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8000)
}

/// True once OUR Flask backend answers on `port`. We don't just check the port is open
/// (a stranger's dev server could hold it) — we GET /status and confirm its shape.
fn backend_ready(port: u16) -> bool {
    use std::io::{Read, Write};
    let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(2)));
    let req =
        format!("GET /status HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    buf.contains("\"phase\"") // it's vr-backend, not whoever else grabbed the port
}

/// Walk up from the working dir until we find app.py (handles `tauri dev` running
/// from src-tauri, and a built exe sitting next to the project).
fn project_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..5 {
        if dir.join("app.py").exists() {
            return dir;
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn spawn_backend(port: u16, model_dir: Option<PathBuf>) -> Option<Child> {
    let root = project_root();
    let py = if cfg!(windows) {
        root.join(".venv/Scripts/python.exe")
    } else {
        root.join(".venv/bin/python")
    };
    let py = if py.exists() { py } else { PathBuf::from("python") };
    let mut cmd = Command::new(py);
    cmd.arg("app.py")
        .current_dir(&root)
        .env("PORT", port.to_string());
    // If the model was bundled into the installer, point Flask at it (no download).
    // In dev this is None and app.py falls back to ./models.
    if let Some(md) = model_dir {
        cmd.env("MODEL_DIR", md);
    }
    cmd.spawn().ok()
}

/// Run the bundled PyInstaller backend (packaged builds). Self-contained — its own
/// Python + torch — so the target machine needs no Python/dev setup.
fn spawn_sidecar(
    exe: &std::path::Path,
    port: u16,
    model_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
) -> Option<Child> {
    let mut cmd = Command::new(exe);
    cmd.env("PORT", port.to_string());
    if let Some(md) = model_dir {
        cmd.env("MODEL_DIR", md);
    }
    if let Some(dd) = data_dir {
        cmd.env("VR_DATA", dd);
    }
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    cmd.spawn().ok()
}

fn kill_backend(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<Backend>() {
        if let Some(mut child) = state.0.lock().unwrap().take() {
            let _ = child.kill();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = free_port();
    tauri::Builder::default()
        // Single-instance MUST be the first plugin: a second launch focuses the existing
        // window and exits, instead of spawning a rival backend that fights for the port
        // and then kills the shared backend when its window closes.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(win) = app.get_webview_window("main") {
                let _ = win.set_focus();
            }
        }))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Backend(Mutex::new(None)))
        .setup(move |app| {
            // Auto-update: on launch, check the GitHub release feed; if a newer signed
            // build exists, download+install and relaunch. Failures (offline, no release
            // yet, dev) are logged and ignored so they never block startup.
            let up = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                use tauri_plugin_updater::UpdaterExt;
                match up.updater() {
                    Ok(updater) => match updater.check().await {
                        Ok(Some(update)) => {
                            eprintln!("updater: installing {}", update.version);
                            // Kill the sidecar FIRST. On Windows the installer replaces the
                            // locked backend/ files then exit(0)s the process WITHOUT firing
                            // RunEvent::Exit, so the on-exit kill below never runs — the old
                            // backend would be orphaned and keep holding the port.
                            kill_backend(&up);
                            if let Err(e) = update.download_and_install(|_, _| {}, || {}).await {
                                eprintln!("updater: install failed: {e}");
                            } else {
                                up.restart();
                            }
                        }
                        Ok(None) => {}
                        Err(e) => eprintln!("updater: check failed: {e}"),
                    },
                    Err(e) => eprintln!("updater: unavailable: {e}"),
                }
            });

            // Look for the bundled backend/model in the installer's resource dir OR next
            // to the exe (portable zip). Neither exists in dev → fall back to venv python.
            let mut roots: Vec<PathBuf> = Vec::new();
            if let Ok(r) = app.path().resource_dir() {
                roots.push(r);
            }
            if let Ok(exe) = std::env::current_exe() {
                if let Some(d) = exe.parent() {
                    roots.push(d.to_path_buf());
                }
            }
            let find = |sub: &str| roots.iter().map(|r| r.join(sub)).find(|p| p.exists());

            let model_dir =
                find("models").filter(|p| p.join("mel_band_roformer_kim_ft_unwa.ckpt").exists());
            let sidecar = find("backend")
                .map(|d| d.join(if cfg!(windows) { "vr-backend.exe" } else { "vr-backend" }))
                .filter(|p| p.exists());
            let data_dir = app.path().app_local_data_dir().ok();

            let child = match sidecar {
                Some(exe) => spawn_sidecar(&exe, port, model_dir, data_dir), // packaged / portable
                None => spawn_backend(port, model_dir),                      // dev: .venv python
            };
            let spawned = child.is_some();
            *app.state::<Backend>().0.lock().unwrap() = child;

            let handle = app.handle().clone();
            // Splash (dist/index.html) shows instantly; once Flask + the model are up,
            // point the window at the real UI served by the backend.
            std::thread::spawn(move || {
                let Some(win) = handle.get_webview_window("main") else {
                    return;
                };
                if !spawned {
                    // Backend process never started (missing exe, AV quarantine).
                    let _ = win.eval(BOOT_FAILED_JS);
                    return;
                }
                let mut ready = false;
                for _ in 0..480 {
                    if backend_ready(port) {
                        ready = true;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
                if ready {
                    if let Ok(url) = tauri::Url::parse(&format!("http://127.0.0.1:{port}/")) {
                        let _ = win.navigate(url);
                    }
                } else {
                    // Don't strand the user on a dead page after 2 minutes — tell them.
                    let _ = win.eval(BOOT_FAILED_JS);
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                kill_backend(app_handle);
            }
        });
}
