//! FILE DRAG-OUT and the macOS SAVE PANEL (photon, Nick 2026-10-26: "drag out? I like" … "we don't want random cache hanging around").
//! The app hands the host a file to drag (its name and a closure that produces its bytes when a drop asks) with [`request`], then returns [`crate::host::EventResponse::StartFileDrag`]; the host cancels the pending click and runs the platform's own drag from the window.
//! No cache: on macOS the drag is a FILE PROMISE (Finder names the folder at drop time and the bytes are written straight there — a user-chosen place, so no privacy prompt); on Windows a VIRTUAL FILE (Explorer pulls the bytes at drop time); on Linux X11 the drag must carry a real path, so the file is staged in `$XDG_RUNTIME_DIR` (per user, in RAM, gone at logout), offered as a MOVE, and removed on the drag's own ending edges.
//! The bytes closure runs once, at drop time, off the UI thread where the platform allows it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// One file to drag out of the window.
#[derive(Clone)]
pub struct DragFile {
    /// The name the drop creates.
    pub name: String,
    /// Its bytes, produced when a drop asks for them (None = the file is not available; the drop fails).
    pub bytes: Arc<dyn Fn() -> Option<Vec<u8>> + Send + Sync>,
}

static PENDING: Mutex<Option<DragFile>> = Mutex::new(None);

/// Name the file the next [`crate::host::EventResponse::StartFileDrag`] drags.
pub fn request(file: DragFile) {
    *PENDING.lock().unwrap() = Some(file);
}

pub(crate) fn take() -> Option<DragFile> {
    PENDING.lock().unwrap().take()
}

/// Write a file's bytes to `path` (a promise's destination on macOS, the staging file on X11).
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn land(file: &DragFile, path: &Path) -> bool {
    match (file.bytes)() {
        Some(b) => match std::fs::write(path, &b) {
            Ok(()) => true,
            Err(e) => {
                log::warn!("FILE-DRAG: writing {} failed: {}", path.display(), e);
                false
            }
        },
        None => {
            log::warn!("FILE-DRAG: {} has no bytes to give", file.name);
            false
        }
    }
}

/// The macOS save panel: the user picks where the file goes (pre-filled with `name`, opening in `dir`), and a place the user picked is writable with no privacy prompt. `None` = cancelled, or not macOS (the app saves where it always does).
pub fn save_dialog(name: &str, dir: Option<&Path>) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        mac::save_dialog(name, dir)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (name, dir);
        None
    }
}

/// Run the platform drag for `file` from `window` (the host calls this on `StartFileDrag`). False when this platform or session cannot drag out.
#[cfg(feature = "host-winit")]
pub(crate) fn start(window: &winit::window::Window, file: DragFile) -> bool {
    #[cfg(target_os = "macos")]
    {
        mac::start(window, file)
    }
    #[cfg(target_os = "windows")]
    {
        let _ = window;
        win::start(file)
    }
    #[cfg(target_os = "linux")]
    {
        x11::start(window, file)
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        let _ = (window, file);
        false
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::DragFile;
    use objc2::rc::Retained;
    use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
    use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass, MainThreadMarker, MainThreadOnly};
    use objc2_app_kit::{NSApplication, NSDragOperation, NSDraggingContext, NSDraggingItem, NSDraggingSession, NSDraggingSource, NSFilePromiseProvider, NSFilePromiseProviderDelegate, NSModalResponseOK, NSSavePanel, NSView, NSWorkspace};
    use objc2_foundation::{NSArray, NSError, NSOperationQueue, NSPoint, NSRect, NSSize, NSString, NSURL};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use std::path::{Path, PathBuf};

    pub(super) struct Ivars {
        file: DragFile,
    }

    define_class!(
        // The promise's delegate and the drag's source in one object; AppKit asks for the file's name on the main thread and writes the promise on the queue we hand it.
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[name = "FluorFileDrag"]
        #[ivars = Ivars]
        struct Source;

        unsafe impl NSObjectProtocol for Source {}

        unsafe impl NSFilePromiseProviderDelegate for Source {
            #[unsafe(method_id(filePromiseProvider:fileNameForType:))]
            fn file_name(&self, _provider: &NSFilePromiseProvider, _file_type: &NSString) -> Retained<NSString> {
                NSString::from_str(&self.ivars().file.name)
            }

            #[unsafe(method(filePromiseProvider:writePromiseToURL:completionHandler:))]
            fn write(&self, _provider: &NSFilePromiseProvider, url: &NSURL, done: &block2::DynBlock<dyn Fn(*mut NSError)>) {
                // The destination Finder chose (a user-chosen place: no privacy prompt); the bytes come from the app now, on our own queue.
                let ok = url.path().map(|p| super::land(&self.ivars().file, std::path::Path::new(&p.to_string()))).unwrap_or(false);
                if ok {
                    done.call((core::ptr::null_mut(),));
                } else {
                    let err = unsafe { NSError::errorWithDomain_code_userInfo(&NSString::from_str("FluorFileDrag"), 1, None) };
                    done.call((Retained::as_ptr(&err) as *mut NSError,));
                }
            }

            #[unsafe(method_id(operationQueueForFilePromiseProvider:))]
            fn queue(&self, _provider: &NSFilePromiseProvider) -> Retained<NSOperationQueue> {
                // Off the main queue: a big file must not freeze the window while it lands.
                NSOperationQueue::new()
            }
        }

        unsafe impl NSDraggingSource for Source {
            #[unsafe(method(draggingSession:sourceOperationMaskForDraggingContext:))]
            fn mask(&self, _session: &NSDraggingSession, context: NSDraggingContext) -> NSDragOperation {
                // Outside the app the file is copied out; inside it, nothing (the app has the file already).
                if context == NSDraggingContext::OutsideApplication { NSDragOperation::Copy } else { NSDragOperation::None }
            }
        }
    );

    pub(super) fn start(window: &winit::window::Window, file: DragFile) -> bool {
        let Some(mtm) = MainThreadMarker::new() else { return false };
        let Ok(handle) = window.window_handle() else { return false };
        let RawWindowHandle::AppKit(h) = handle.as_raw() else { return false };
        // SAFETY: winit's AppKit handle names its live content view, and we are on the main thread.
        let view: &NSView = unsafe { h.ns_view.cast::<NSView>().as_ref() };
        let app = NSApplication::sharedApplication(mtm);
        // The drag rides the event being handled (the mouse-dragged that crossed the threshold).
        let Some(event) = app.currentEvent() else { return false };
        let ext = std::path::Path::new(&file.name).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let source: Retained<Source> = {
            let this = Source::alloc(mtm).set_ivars(Ivars { file });
            unsafe { msg_send![super(this), init] }
        };
        let provider = NSFilePromiseProvider::initWithFileType_delegate(NSFilePromiseProvider::alloc(), &NSString::from_str("public.data"), ProtocolObject::from_ref(&*source));
        let item = NSDraggingItem::initWithPasteboardWriter(NSDraggingItem::alloc(), ProtocolObject::from_ref(&*provider));
        // The file's own icon under the pointer (iconForFileType is deprecated for iconForContentType, which needs a UTType from another crate; the old call still answers on every macOS photon runs on).
        #[allow(deprecated)]
        let icon = NSWorkspace::sharedWorkspace().iconForFileType(&NSString::from_str(&ext));
        let side = 64.0;
        icon.setSize(NSSize::new(side, side));
        let at = view.convertPoint_fromView(event.locationInWindow(), None);
        unsafe { item.setDraggingFrame_contents(NSRect::new(NSPoint::new(at.x - side / 2.0, at.y - side / 2.0), NSSize::new(side, side)), Some(&icon)) };
        let items = NSArray::from_retained_slice(&[item]);
        view.beginDraggingSessionWithItems_event_source(&items, &event, ProtocolObject::from_ref(&*source));
        // The promise provider holds its delegate WEAKLY, and Finder may ask for the file after the drag session itself has ended — so the source lives for the process (one small object per drag, never a file).
        core::mem::forget(source);
        core::mem::forget(provider);
        true
    }

    pub(super) fn save_dialog(name: &str, dir: Option<&Path>) -> Option<PathBuf> {
        let mtm = MainThreadMarker::new()?;
        let panel = NSSavePanel::savePanel(mtm);
        panel.setNameFieldStringValue(&NSString::from_str(name));
        if let Some(d) = dir {
            panel.setDirectoryURL(Some(&NSURL::fileURLWithPath(&NSString::from_str(&d.to_string_lossy()))));
        }
        if panel.runModal() != NSModalResponseOK {
            return None;
        }
        Some(PathBuf::from(panel.URL()?.path()?.to_string()))
    }
}

#[cfg(target_os = "windows")]
mod win {
    //! A VIRTUAL FILE: the drag offers the file's descriptor and, when Explorer drops it, its contents — produced then, from the app, so no file exists anywhere until the drop writes it.
    use super::DragFile;
    use std::cell::Cell;
    use windows::core::{implement, Interface, Result, HRESULT};
    use windows::Win32::Foundation::{BOOL, DRAGDROP_S_CANCEL, DRAGDROP_S_DROP, DRAGDROP_S_USEDEFAULTCURSORS, DV_E_FORMATETC, E_NOTIMPL, HGLOBAL, OLE_E_ADVISENOTSUPPORTED, S_OK};
    use windows::Win32::System::Com::{IAdviseSink, IDataObject, IDataObject_Impl, IEnumFORMATETC, IEnumSTATDATA, DVASPECT_CONTENT, FORMATETC, STGMEDIUM, STGMEDIUM_0, TYMED_HGLOBAL};
    use windows::Win32::System::DataExchange::RegisterClipboardFormatW;
    use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
    use windows::Win32::System::Ole::{DoDragDrop, IDropSource, IDropSource_Impl, OleInitialize, DROPEFFECT, DROPEFFECT_COPY};
    use windows::Win32::System::SystemServices::{MK_LBUTTON, MODIFIERKEYS_FLAGS};
    use windows::Win32::UI::Shell::{SHCreateStdEnumFmtEtc, CFSTR_FILECONTENTS, CFSTR_FILEDESCRIPTORW, FD_FILESIZE, FD_PROGRESSUI, FILEDESCRIPTORW, FILEGROUPDESCRIPTORW};

    #[implement(IDataObject, IDropSource)]
    struct Drag {
        file: DragFile,
        descriptor: u16,
        contents: u16,
        /// The bytes, produced on the first ask (Explorer asks for the size in the descriptor before it asks for the contents).
        bytes: Cell<Option<Option<Vec<u8>>>>,
    }

    impl Drag {
        fn bytes(&self) -> Option<Vec<u8>> {
            let b = self.bytes.take().unwrap_or_else(|| (self.file.bytes)());
            self.bytes.set(Some(b.clone()));
            b
        }

        fn formats(&self) -> [FORMATETC; 2] {
            let f = |cf: u16, lindex: i32| FORMATETC { cfFormat: cf, ptd: core::ptr::null_mut(), dwAspect: DVASPECT_CONTENT.0, lindex, tymed: TYMED_HGLOBAL.0 as u32 };
            [f(self.descriptor, -1), f(self.contents, 0)]
        }
    }

    /// Bytes into a movable global the receiver owns once handed over.
    fn global(bytes: &[u8]) -> Result<HGLOBAL> {
        unsafe {
            let h = GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1))?;
            let p = GlobalLock(h) as *mut u8;
            if !p.is_null() {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
            }
            let _ = GlobalUnlock(h);
            Ok(h)
        }
    }

    fn medium(h: HGLOBAL) -> STGMEDIUM {
        STGMEDIUM { tymed: TYMED_HGLOBAL.0 as u32, u: STGMEDIUM_0 { hGlobal: h }, pUnkForRelease: core::mem::ManuallyDrop::new(None) }
    }

    impl IDataObject_Impl for Drag_Impl {
        fn GetData(&self, pformatetcin: *const FORMATETC) -> Result<STGMEDIUM> {
            // SAFETY: OLE hands a valid FORMATETC for the call's duration.
            let f = unsafe { &*pformatetcin };
            if f.tymed & TYMED_HGLOBAL.0 as u32 == 0 {
                return Err(DV_E_FORMATETC.into());
            }
            if f.cfFormat == self.descriptor {
                let mut d = FILEDESCRIPTORW { dwFlags: (FD_FILESIZE.0 | FD_PROGRESSUI.0) as u32, ..Default::default() };
                let size = self.bytes().map_or(0, |b| b.len() as u64);
                d.nFileSizeHigh = (size >> 32) as u32;
                d.nFileSizeLow = size as u32;
                // The name, truncated to the descriptor's 259 UTF-16 units plus the terminator.
                for (i, u) in self.file.name.encode_utf16().take(259).enumerate() {
                    d.cFileName[i] = u;
                }
                let group = FILEGROUPDESCRIPTORW { cItems: 1, fgd: [d] };
                // SAFETY: FILEGROUPDESCRIPTORW is plain old data; its bytes are what the receiver reads back.
                let raw = unsafe { core::slice::from_raw_parts(&group as *const _ as *const u8, core::mem::size_of::<FILEGROUPDESCRIPTORW>()) };
                return Ok(medium(global(raw)?));
            }
            if f.cfFormat == self.contents && f.lindex <= 0 {
                let Some(b) = self.bytes() else { return Err(DV_E_FORMATETC.into()) };
                return Ok(medium(global(&b)?));
            }
            Err(DV_E_FORMATETC.into())
        }

        fn GetDataHere(&self, _f: *const FORMATETC, _m: *mut STGMEDIUM) -> Result<()> {
            Err(E_NOTIMPL.into())
        }

        fn QueryGetData(&self, pformatetc: *const FORMATETC) -> HRESULT {
            // SAFETY: as in GetData.
            let f = unsafe { &*pformatetc };
            if (f.cfFormat == self.descriptor || f.cfFormat == self.contents) && f.tymed & TYMED_HGLOBAL.0 as u32 != 0 { S_OK } else { DV_E_FORMATETC }
        }

        fn GetCanonicalFormatEtc(&self, _i: *const FORMATETC, _o: *mut FORMATETC) -> HRESULT {
            E_NOTIMPL
        }

        fn SetData(&self, _f: *const FORMATETC, _m: *const STGMEDIUM, _r: BOOL) -> Result<()> {
            // Explorer reports its performed drop effect here; nothing of ours depends on it.
            Err(E_NOTIMPL.into())
        }

        fn EnumFormatEtc(&self, _direction: u32) -> Result<IEnumFORMATETC> {
            unsafe { SHCreateStdEnumFmtEtc(&self.formats()) }
        }

        fn DAdvise(&self, _f: *const FORMATETC, _a: u32, _s: Option<&IAdviseSink>) -> Result<u32> {
            Err(OLE_E_ADVISENOTSUPPORTED.into())
        }

        fn DUnadvise(&self, _c: u32) -> Result<()> {
            Err(OLE_E_ADVISENOTSUPPORTED.into())
        }

        fn EnumDAdvise(&self) -> Result<IEnumSTATDATA> {
            Err(OLE_E_ADVISENOTSUPPORTED.into())
        }
    }

    impl IDropSource_Impl for Drag_Impl {
        fn QueryContinueDrag(&self, escape: BOOL, keys: MODIFIERKEYS_FLAGS) -> HRESULT {
            if escape.as_bool() {
                DRAGDROP_S_CANCEL
            } else if keys.0 & MK_LBUTTON.0 == 0 {
                DRAGDROP_S_DROP
            } else {
                S_OK
            }
        }

        fn GiveFeedback(&self, _effect: DROPEFFECT) -> HRESULT {
            DRAGDROP_S_USEDEFAULTCURSORS
        }
    }

    pub(super) fn start(file: DragFile) -> bool {
        unsafe {
            // OLE on this (the UI) thread: a no-op when winit already initialised it; a thread already in the multithreaded apartment cannot drag.
            if let Err(e) = OleInitialize(None) {
                log::warn!("FILE-DRAG: OLE unavailable on the UI thread: {}", e);
                return false;
            }
            let descriptor = RegisterClipboardFormatW(CFSTR_FILEDESCRIPTORW) as u16;
            let contents = RegisterClipboardFormatW(CFSTR_FILECONTENTS) as u16;
            let name = file.name.clone();
            let drag = Drag { file, descriptor, contents, bytes: Cell::new(None) };
            let data: IDataObject = drag.into();
            let source: IDropSource = data.cast().expect("one object implements both");
            let mut effect = DROPEFFECT(0);
            // The OS's own modal drag loop: it owns the mouse until the drop or the cancel, then returns here.
            let hr = DoDragDrop(&data, &source, DROPEFFECT_COPY, &mut effect);
            log::info!("FILE-DRAG: {} → {}", name, if hr == DRAGDROP_S_DROP { "dropped" } else { "cancelled" });
        }
        true
    }
}

#[cfg(target_os = "linux")]
mod x11 {
    //! An XDND SOURCE on its own connection and thread (winit receives drops but cannot start a drag). The drag must carry a real path, so the file is staged in `$XDG_RUNTIME_DIR` (per user, in RAM, gone at logout) and offered as a MOVE; it is removed on the drag's own edges — a cancel, a drop nobody accepted, the target's DELETE request — and a move takes it with it. A target that only copies leaves the staged file for its own asynchronous copy to read; the next drag's sweep (and logout) clears it.
    //! While the button is down, X gives the window it was pressed in an implicit pointer grab, held by winit's connection; ours cannot grab until it is released, so it is released thru winit's own display first.
    use super::DragFile;
    use std::path::{Path, PathBuf};
    use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{Atom, AtomEnum, ClientMessageEvent, ConnectionExt as _, CreateWindowAux, EventMask, GrabMode, GrabStatus, PropMode, SelectionNotifyEvent, SelectionRequestEvent, Window, WindowClass, SELECTION_NOTIFY_EVENT};
    use x11rb::protocol::Event;
    use x11rb::rust_connection::RustConnection;
    use x11rb::wrapper::ConnectionExt as _;
    use x11rb::{COPY_DEPTH_FROM_PARENT, COPY_FROM_PARENT, CURRENT_TIME, NONE};

    /// The protocol version we speak (XDND 5: XdndFinished carries the performed action).
    const XDND_VERSION: u32 = 5;
    /// The cursor font's hand glyph.
    const XC_HAND2: u16 = 60;

    struct Atoms {
        aware: Atom,
        proxy: Atom,
        selection: Atom,
        enter: Atom,
        position: Atom,
        status: Atom,
        leave: Atom,
        drop: Atom,
        finished: Atom,
        action_move: Atom,
        uri_list: Atom,
        targets: Atom,
        delete: Atom,
    }

    fn atoms(c: &RustConnection) -> Result<Atoms, Box<dyn std::error::Error>> {
        let names: [&[u8]; 13] = [b"XdndAware", b"XdndProxy", b"XdndSelection", b"XdndEnter", b"XdndPosition", b"XdndStatus", b"XdndLeave", b"XdndDrop", b"XdndFinished", b"XdndActionMove", b"text/uri-list", b"TARGETS", b"DELETE"];
        let cookies: Vec<_> = names.iter().map(|n| c.intern_atom(false, n)).collect::<Result<_, _>>()?;
        let mut a = Vec::with_capacity(13);
        for k in cookies {
            a.push(k.reply()?.atom);
        }
        Ok(Atoms { aware: a[0], proxy: a[1], selection: a[2], enter: a[3], position: a[4], status: a[5], leave: a[6], drop: a[7], finished: a[8], action_move: a[9], uri_list: a[10], targets: a[11], delete: a[12] })
    }

    /// The staging folder: `$XDG_RUNTIME_DIR/fluor-drag` (RAM, this user's only), else the temp folder.
    fn staging_dir() -> PathBuf {
        std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join("fluor-drag")
    }

    /// Clear anything an earlier drag left (a copy-only target's file, a drag cut off by a crash).
    pub(super) fn sweep() {
        if let Ok(rd) = std::fs::read_dir(staging_dir()) {
            for e in rd.flatten() {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    fn stage(file: &DragFile) -> Option<PathBuf> {
        use std::os::unix::fs::DirBuilderExt;
        sweep();
        let dir = staging_dir();
        let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir);
        // A name is one path component: no separators, no NUL.
        let name: String = file.name.chars().map(|c| if c == '/' || c == '\0' { '_' } else { c }).collect();
        let path = dir.join(if name.is_empty() { "file".into() } else { name });
        super::land(file, &path).then_some(path)
    }

    /// `file://` + the path percent-encoded, the uri-list line ending.
    fn uri(path: &Path) -> Vec<u8> {
        use std::os::unix::ffi::OsStrExt;
        let mut out = b"file://".to_vec();
        for &b in path.as_os_str().as_bytes() {
            if b.is_ascii_alphanumeric() || b"/-_.~".contains(&b) {
                out.push(b);
            } else {
                out.extend_from_slice(format!("%{:02X}", b).as_bytes());
            }
        }
        out.extend_from_slice(b"\r\n");
        out
    }

    /// Release the implicit grab winit's connection holds while the button is down.
    fn release_implicit_grab(window: &winit::window::Window) {
        use core::ffi::{c_int, c_uint, c_ulong, c_void};
        let Ok(d) = window.display_handle() else { return };
        // SAFETY: the libraries are the ones winit already loaded (RTLD_NOLOAD finds them, never loads anew); each symbol is called with the signature its header declares, on the display / connection winit handed out.
        unsafe {
            match d.as_raw() {
                RawDisplayHandle::Xlib(h) => {
                    let Some(dpy) = h.display else { return };
                    let lib = libc::dlopen(c"libX11.so.6".as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
                    if lib.is_null() {
                        return;
                    }
                    let (ungrab, flush) = (libc::dlsym(lib, c"XUngrabPointer".as_ptr()), libc::dlsym(lib, c"XFlush".as_ptr()));
                    if !ungrab.is_null() && !flush.is_null() {
                        let ungrab: extern "C" fn(*mut c_void, c_ulong) -> c_int = core::mem::transmute(ungrab);
                        let flush: extern "C" fn(*mut c_void) -> c_int = core::mem::transmute(flush);
                        ungrab(dpy.as_ptr(), 0);
                        flush(dpy.as_ptr());
                    }
                }
                RawDisplayHandle::Xcb(h) => {
                    let Some(conn) = h.connection else { return };
                    let lib = libc::dlopen(c"libxcb.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
                    if lib.is_null() {
                        return;
                    }
                    let (ungrab, flush) = (libc::dlsym(lib, c"xcb_ungrab_pointer".as_ptr()), libc::dlsym(lib, c"xcb_flush".as_ptr()));
                    if !ungrab.is_null() && !flush.is_null() {
                        let ungrab: extern "C" fn(*mut c_void, u32) -> c_uint = core::mem::transmute(ungrab);
                        let flush: extern "C" fn(*mut c_void) -> c_int = core::mem::transmute(flush);
                        ungrab(conn.as_ptr(), 0);
                        flush(conn.as_ptr());
                    }
                }
                _ => {}
            }
        }
    }

    pub(super) fn start(window: &winit::window::Window, file: DragFile) -> bool {
        let Ok(handle) = window.window_handle() else { return false };
        let own = match handle.as_raw() {
            RawWindowHandle::Xcb(h) => h.window.get(),
            RawWindowHandle::Xlib(h) => h.window as u32,
            _ => return false,
        };
        let Some(path) = stage(&file) else { return false };
        release_implicit_grab(window);
        let name = file.name.clone();
        let spawned = std::thread::Builder::new().name("fluor-xdnd".into()).spawn(move || {
            match run(own, &path) {
                Ok(outcome) => log::info!("FILE-DRAG: {} → {}", name, outcome),
                Err(e) => log::warn!("FILE-DRAG: {} failed: {}", name, e),
            }
        });
        spawned.is_ok()
    }

    /// The XdndAware window under the pointer (or the proxy it names) and its version; never our own window.
    fn target_at(c: &RustConnection, root: Window, own: Window, a: &Atoms) -> Result<Option<(Window, Window, u32)>, Box<dyn std::error::Error>> {
        let mut w = root;
        loop {
            let child = c.query_pointer(w)?.reply()?.child;
            if child == NONE || child == own {
                return Ok(None);
            }
            let aware = c.get_property(false, child, a.aware, AtomEnum::ATOM, 0, 1)?.reply()?;
            if let Some(v) = aware.value32().and_then(|mut i| i.next()) {
                // Messages go to the proxy the window names, if any.
                let proxy = c.get_property(false, child, a.proxy, AtomEnum::WINDOW, 0, 1)?.reply()?.value32().and_then(|mut i| i.next()).unwrap_or(child);
                return Ok(Some((child, proxy, v)));
            }
            w = child;
        }
    }

    fn send(c: &RustConnection, to: Window, kind: Atom, data: [u32; 5]) -> Result<(), Box<dyn std::error::Error>> {
        c.send_event(false, to, EventMask::NO_EVENT, ClientMessageEvent::new(32, to, kind, data))?;
        Ok(())
    }

    /// Answer the target's request for the selection: the file's uri, the targets we offer, or DELETE (a move's last step: the target has the file, ours goes).
    fn answer(c: &RustConnection, r: &SelectionRequestEvent, a: &Atoms, uri: &[u8], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let prop = if r.property == NONE { r.target } else { r.property };
        let ok = if r.target == a.uri_list {
            c.change_property8(PropMode::REPLACE, r.requestor, prop, a.uri_list, uri)?;
            true
        } else if r.target == a.targets {
            c.change_property32(PropMode::REPLACE, r.requestor, prop, AtomEnum::ATOM, &[a.targets, a.uri_list, a.delete])?;
            true
        } else if r.target == a.delete {
            let _ = std::fs::remove_file(path);
            c.change_property8(PropMode::REPLACE, r.requestor, prop, AtomEnum::NONE, &[])?;
            true
        } else {
            false
        };
        let notify = SelectionNotifyEvent { response_type: SELECTION_NOTIFY_EVENT, sequence: 0, time: r.time, requestor: r.requestor, selection: r.selection, target: r.target, property: if ok { prop } else { NONE } };
        c.send_event(false, r.requestor, EventMask::NO_EVENT, notify)?;
        c.flush()?;
        Ok(())
    }

    /// The drag, from the grab to the target's finish (or the cancel). Returns what happened, for the log.
    fn run(own: Window, path: &Path) -> Result<&'static str, Box<dyn std::error::Error>> {
        let (c, screen) = x11rb::connect(None)?;
        let root = c.setup().roots[screen].root;
        let a = atoms(&c)?;
        let uri = uri(path);
        let win = c.generate_id()?;
        c.create_window(COPY_DEPTH_FROM_PARENT, win, root, -10, -10, 1, 1, 0, WindowClass::INPUT_ONLY, COPY_FROM_PARENT, &CreateWindowAux::new())?;
        c.set_selection_owner(win, a.selection, CURRENT_TIME)?;
        let font = c.generate_id()?;
        c.open_font(font, b"cursor")?;
        let cursor = c.generate_id()?;
        c.create_glyph_cursor(cursor, font, font, XC_HAND2, XC_HAND2 + 1, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF)?;
        let grab = c.grab_pointer(false, win, EventMask::POINTER_MOTION | EventMask::BUTTON_RELEASE, GrabMode::ASYNC, GrabMode::ASYNC, NONE, cursor, CURRENT_TIME)?.reply()?;
        let done = |c: &RustConnection| -> Result<(), Box<dyn std::error::Error>> {
            c.ungrab_pointer(CURRENT_TIME)?;
            c.free_cursor(cursor)?;
            c.close_font(font)?;
            c.destroy_window(win)?;
            c.flush()?;
            Ok(())
        };
        if grab.status != GrabStatus::SUCCESS {
            done(&c)?;
            let _ = std::fs::remove_file(path);
            return Err(format!("pointer grab refused ({:?})", grab.status).into());
        }
        c.flush()?;
        // (the window under the pointer, where its messages go, its version)
        let mut target: Option<(Window, Window, u32)> = None;
        let mut accepted = false;
        let mut dropped = false;
        let outcome = loop {
            match c.wait_for_event()? {
                Event::MotionNotify(m) if !dropped => {
                    let now = target_at(&c, root, own, &a)?;
                    if now.map(|t| t.0) != target.map(|t| t.0) {
                        if let Some((_, to, _)) = target {
                            send(&c, to, a.leave, [win, 0, 0, 0, 0])?;
                        }
                        accepted = false;
                        if let Some((_, to, v)) = now {
                            send(&c, to, a.enter, [win, v.min(XDND_VERSION) << 24, a.uri_list, 0, 0])?;
                        }
                        target = now;
                    }
                    if let Some((_, to, _)) = target {
                        // WHY/PROOF: root coordinates are i16 from the server and XDND packs each into 16 bits; `as u16` keeps the bit pattern the protocol expects.
                        send(&c, to, a.position, [win, 0, ((m.root_x as u16 as u32) << 16) | m.root_y as u16 as u32, m.time, a.action_move])?;
                    }
                    c.flush()?;
                }
                Event::ClientMessage(m) if m.type_ == a.status => {
                    accepted = m.data.as_data32()[1] & 1 != 0;
                }
                Event::ButtonRelease(b) if !dropped => match target {
                    Some((_, to, _)) if accepted => {
                        send(&c, to, a.drop, [win, 0, b.time, 0, 0])?;
                        c.ungrab_pointer(CURRENT_TIME)?;
                        c.flush()?;
                        dropped = true;
                    }
                    _ => {
                        if let Some((_, to, _)) = target {
                            send(&c, to, a.leave, [win, 0, 0, 0, 0])?;
                        }
                        let _ = std::fs::remove_file(path);
                        break "cancelled";
                    }
                },
                Event::SelectionRequest(r) => answer(&c, &r, &a, &uri, path)?,
                Event::ClientMessage(m) if m.type_ == a.finished => break "dropped",
                // Another drag took the selection: ours is over.
                Event::SelectionClear(_) => break "superseded",
                _ => {}
            }
        };
        done(&c)?;
        Ok(outcome)
    }
}
