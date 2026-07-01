use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

// Holds the Python (Flask) child process so we can kill it when the app exits.
struct Backend(Mutex<Option<Child>>);

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

fn spawn_backend(model_dir: Option<PathBuf>) -> Option<Child> {
    let root = project_root();
    let py = if cfg!(windows) {
        root.join(".venv/Scripts/python.exe")
    } else {
        root.join(".venv/bin/python")
    };
    let py = if py.exists() { py } else { PathBuf::from("python") };
    let mut cmd = Command::new(py);
    cmd.arg("app.py").current_dir(&root).env("PORT", "8000");
    // If the model was bundled into the installer, point Flask at it (no download).
    // In dev this is None and app.py falls back to ./models.
    if let Some(md) = model_dir {
        cmd.env("MODEL_DIR", md);
    }
    cmd.spawn().ok()
}

/// Run the bundled PyInstaller backend (packaged builds). Self-contained — its own
/// Python + torch + CUDA — so the target machine needs no Python/dev setup.
fn spawn_sidecar(
    exe: &std::path::Path,
    model_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
) -> Option<Child> {
    let mut cmd = Command::new(exe);
    cmd.env("PORT", "8000");
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Backend(Mutex::new(None)))
        .setup(|app| {
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
                Some(exe) => spawn_sidecar(&exe, model_dir, data_dir), // packaged / portable
                None => spawn_backend(model_dir),                      // dev: .venv python app.py
            };
            *app.state::<Backend>().0.lock().unwrap() = child;

            let handle = app.handle().clone();
            // Splash (dist/index.html) shows instantly; once Flask + the model are
            // up, point the window at the real UI served by the backend.
            std::thread::spawn(move || {
                for _ in 0..480 {
                    if std::net::TcpStream::connect("127.0.0.1:8000").is_ok() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
                if let Some(win) = handle.get_webview_window("main") {
                    if let Ok(url) = tauri::Url::parse("http://127.0.0.1:8000/") {
                        let _ = win.navigate(url);
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app_handle.try_state::<Backend>() {
                    if let Some(mut child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
