//! `font_pack <pack id> <font file> <source> <licence file> <out.vsf>`: one unmodified font file made a signed font pack ([`fluor::packs`]), then read back through the same check an app makes.
//! The key is the author's, read from FLUOR_SIGNING_KEY or /mnt/Harbor/Code/keys/photon-signing-key (32 raw bytes) and never printed.
fn fail(msg: &str) -> ! {
    eprintln!("font_pack: {msg}");
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [id, font, source, licence, out] = args.as_slice() else { fail("usage: font_pack <pack id> <font file> <source> <licence file> <out.vsf>") };
    let path = std::env::var("FLUOR_SIGNING_KEY").unwrap_or_else(|_| "/mnt/Harbor/Code/keys/photon-signing-key".into());
    let key: [u8; 32] = std::fs::read(&path).unwrap_or_else(|e| fail(&format!("{path}: {e}"))).try_into().unwrap_or_else(|_| fail("the signing key must be exactly 32 bytes"));
    let face = std::fs::read(font).unwrap_or_else(|e| fail(&format!("{font}: {e}")));
    let licence = std::fs::read_to_string(licence).unwrap_or_else(|e| fail(&format!("{licence}: {e}")));
    let n = face.len();
    let doc = fluor::packs::write(id, face.clone(), source, &licence, &key).unwrap_or_else(|e| fail(&e));
    let author = ed25519_dalek::SigningKey::from_bytes(&key).verifying_key().to_bytes();
    let (i, back) = fluor::packs::read(&doc, author).unwrap_or_else(|e| fail(&format!("does not read back: {e}")));
    if back != face || fluor::packs::PACKS[i].id != id {
        fail("the face does not read back unchanged");
    }
    let mut t = fluor::text::TextRenderer::new();
    t.load_pack(&doc, author).unwrap_or_else(|e| fail(&format!("does not load: {e}")));
    std::fs::write(out, &doc).unwrap_or_else(|e| fail(&format!("{out}: {e}")));
    println!("{id}: {} KB face, {} KB pack, {}", n / 1024, doc.len() / 1024, fluor::packs::PACKS[i].family);
}
