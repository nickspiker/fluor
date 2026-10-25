//! `fluor_gpu`: the targets whose present goes thru wgpu — macOS always, Linux with the `gpu` feature.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(fluor_gpu)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some();
    if os == "macos" || (os == "linux" && gpu) {
        println!("cargo::rustc-cfg=fluor_gpu");
    }
}
