//! Font packs (Nick 2026-10-16): the faces fluor does not carry in the binary, fetched by the app the first time text needs one, each a signed VSF document holding one unmodified font file.
//!
//! The binary stays small and every script still draws native: text in a script with no face loaded draws as it always has (the fallback chain, tofu at worst), the renderer notes which pack it needs, the app fetches that pack, [`crate::text::TextRenderer::load_pack`] checks the author's signature and loads the face, and the next frame draws the text in its own script.
//! A pack is a document with one `font` section: `pack` (the id here), `family` (the face's family name, checked against the face), `source` (where the face came from), `licence` (its licence text) and `face` (the font file's bytes, a `u8` tensor).
//! The faces are unmodified upstream files (Google Fonts' Noto): no subsetting, no hinting dropped.
//! Chinese, Japanese and Korean share code points, each with its own regional glyph forms: kana in a text picks Japanese, Hangul picks Korean, and Han alone takes the renderer's default (Simplified Chinese unless the app says otherwise, [`crate::text::TextRenderer::set_han`]).

/// One pack: its id (its file is `<id>.vsf` wherever the app fetches packs from) and the family its face declares.
#[derive(Debug, PartialEq, Eq)]
pub struct Pack {
    pub id: &'static str,
    pub family: &'static str,
}

/// Every pack, in a fixed order (a renderer keeps which are loaded as bits by index).
pub const PACKS: &[Pack] = &[
    Pack { id: "noto-sans-arabic", family: "Noto Sans Arabic" },
    Pack { id: "noto-sans-armenian", family: "Noto Sans Armenian" },
    Pack { id: "noto-sans-bengali", family: "Noto Sans Bengali" },
    Pack { id: "noto-sans-canadian-aboriginal", family: "Noto Sans Canadian Aboriginal" },
    Pack { id: "noto-sans-cherokee", family: "Noto Sans Cherokee" },
    Pack { id: "noto-sans-devanagari", family: "Noto Sans Devanagari" },
    Pack { id: "noto-sans-ethiopic", family: "Noto Sans Ethiopic" },
    Pack { id: "noto-sans-georgian", family: "Noto Sans Georgian" },
    Pack { id: "noto-sans-gujarati", family: "Noto Sans Gujarati" },
    Pack { id: "noto-sans-gurmukhi", family: "Noto Sans Gurmukhi" },
    Pack { id: "noto-sans-hebrew", family: "Noto Sans Hebrew" },
    Pack { id: "noto-sans-kannada", family: "Noto Sans Kannada" },
    Pack { id: "noto-sans-khmer", family: "Noto Sans Khmer" },
    Pack { id: "noto-sans-lao", family: "Noto Sans Lao" },
    Pack { id: "noto-sans-malayalam", family: "Noto Sans Malayalam" },
    Pack { id: "noto-sans-mongolian", family: "Noto Sans Mongolian" },
    Pack { id: "noto-sans-myanmar", family: "Noto Sans Myanmar" },
    Pack { id: "noto-sans-oriya", family: "Noto Sans Oriya" },
    Pack { id: "noto-sans-runic", family: "Noto Sans Runic" },
    Pack { id: "noto-sans-sinhala", family: "Noto Sans Sinhala" },
    Pack { id: "noto-sans-syriac", family: "Noto Sans Syriac" },
    Pack { id: "noto-sans-tamil", family: "Noto Sans Tamil" },
    Pack { id: "noto-sans-telugu", family: "Noto Sans Telugu" },
    Pack { id: "noto-sans-thaana", family: "Noto Sans Thaana" },
    Pack { id: "noto-sans-thai", family: "Noto Sans Thai" },
    Pack { id: "noto-sans-tifinagh", family: "Noto Sans Tifinagh" },
    Pack { id: "noto-serif-tibetan", family: "Noto Serif Tibetan" },
    Pack { id: "noto-sans-sc", family: "Noto Sans SC" },
    Pack { id: "noto-sans-tc", family: "Noto Sans TC" },
    Pack { id: "noto-sans-hk", family: "Noto Sans HK" },
    Pack { id: "noto-sans-jp", family: "Noto Sans JP" },
    Pack { id: "noto-sans-kr", family: "Noto Sans KR" },
];

const _: () = assert!(PACKS.len() <= 64, "a renderer keeps packs as bits of a u64");

/// The regional faces for Chinese, Japanese and Korean text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Han {
    Simplified,
    Traditional,
    HongKong,
    Japanese,
    Korean,
}

impl Han {
    /// The pack drawing this region's forms.
    pub fn pack(self) -> usize {
        index(match self {
            Han::Simplified => "noto-sans-sc",
            Han::Traditional => "noto-sans-tc",
            Han::HongKong => "noto-sans-hk",
            Han::Japanese => "noto-sans-jp",
            Han::Korean => "noto-sans-kr",
        })
        .expect("every region has a pack")
    }
}

/// A pack's index by id.
pub fn index(id: &str) -> Option<usize> {
    PACKS.iter().position(|p| p.id == id)
}

/// What draws a character outside what the binary carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// The pack for its script.
    Script(usize),
    /// Kana: Japanese.
    Kana,
    /// Hangul: Korean.
    Hangul,
    /// Han, CJK punctuation, full-width forms and the rest the regions share.
    Shared,
}

/// Which pack a character needs, if any (None for everything the binary's own faces draw: Latin, Greek, Cyrillic, symbols, emoji).
pub fn need(c: char) -> Option<Need> {
    let script = |id: &str| index(id).map(Need::Script);
    match c as u32 {
        0..=0x052F => None,
        0x0530..=0x058F | 0xFB13..=0xFB17 => script("noto-sans-armenian"),
        0x0590..=0x05FF | 0xFB1D..=0xFB4F => script("noto-sans-hebrew"),
        0x0600..=0x06FF | 0x0750..=0x077F | 0x0870..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => script("noto-sans-arabic"),
        0x0700..=0x074F | 0x0860..=0x086F => script("noto-sans-syriac"),
        0x0780..=0x07BF => script("noto-sans-thaana"),
        0x0900..=0x097F | 0xA8E0..=0xA8FF => script("noto-sans-devanagari"),
        0x0980..=0x09FF => script("noto-sans-bengali"),
        0x0A00..=0x0A7F => script("noto-sans-gurmukhi"),
        0x0A80..=0x0AFF => script("noto-sans-gujarati"),
        0x0B00..=0x0B7F => script("noto-sans-oriya"),
        0x0B80..=0x0BFF => script("noto-sans-tamil"),
        0x0C00..=0x0C7F => script("noto-sans-telugu"),
        0x0C80..=0x0CFF => script("noto-sans-kannada"),
        0x0D00..=0x0D7F => script("noto-sans-malayalam"),
        0x0D80..=0x0DFF => script("noto-sans-sinhala"),
        0x0E00..=0x0E7F => script("noto-sans-thai"),
        0x0E80..=0x0EFF => script("noto-sans-lao"),
        0x0F00..=0x0FFF => script("noto-serif-tibetan"),
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => script("noto-sans-myanmar"),
        0x10A0..=0x10FF | 0x1C90..=0x1CBF | 0x2D00..=0x2D2F => script("noto-sans-georgian"),
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7FF => Some(Need::Hangul),
        0x1200..=0x139F | 0x2D80..=0x2DDF | 0xAB00..=0xAB2F => script("noto-sans-ethiopic"),
        0x13A0..=0x13FF | 0xAB70..=0xABBF => script("noto-sans-cherokee"),
        0x1400..=0x167F | 0x18B0..=0x18FF => script("noto-sans-canadian-aboriginal"),
        0x16A0..=0x16FF => script("noto-sans-runic"),
        0x1780..=0x17FF | 0x19E0..=0x19FF => script("noto-sans-khmer"),
        0x1800..=0x18AF => script("noto-sans-mongolian"),
        0x2D30..=0x2D7F => script("noto-sans-tifinagh"),
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0x1B000..=0x1B16F => Some(Need::Kana),
        0x2E80..=0x2FDF | 0x3000..=0x303F | 0x3100..=0x312F | 0x31A0..=0x31EF | 0x3200..=0x33FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFFEF | 0x20000..=0x3134F => Some(Need::Shared),
        _ => None,
    }
}

/// Whether a character draws from a Chinese, Japanese or Korean face.
pub fn is_cjk(c: char) -> bool {
    matches!(need(c), Some(Need::Kana | Need::Hangul | Need::Shared))
}

/// A pack document's face, read after its signature is checked against the author's key: the pack and the font file's bytes. Nothing unsigned, signed by another key, or naming a pack not in [`PACKS`] is accepted.
#[cfg(feature = "packs")]
pub fn read(doc: &[u8], author: [u8; 32]) -> Result<(usize, Vec<u8>), String> {
    use vsf::VsfType;
    let (header, end) = vsf::verification::read_verified(doc, Some(author)).map_err(|e| format!("font pack not signed by the author: {e}"))?;
    let sections = header.sections(doc, end).map_err(|e| format!("font pack sections: {e}"))?;
    let s = sections.iter().find(|s| s.name == "font").ok_or("a font pack without its font section")?;
    let first = |name: &str| s.get_fields(name).first().and_then(|f| f.values.first()).cloned();
    let Some(VsfType::x(id)) = first("pack") else { return Err("a font pack without its id".into()) };
    let i = index(&id).ok_or_else(|| format!("unknown font pack {id}"))?;
    let Some(VsfType::x(family)) = first("family") else { return Err("a font pack without its family".into()) };
    if family != PACKS[i].family {
        return Err(format!("font pack {id} names {family}, not {}", PACKS[i].family));
    }
    match first("face") {
        Some(VsfType::t_u3(t)) if !t.data.is_empty() => Ok((i, t.data)),
        _ => Err(format!("font pack {id} without its face")),
    }
}

/// A pack document (the publishing side): one unmodified font file, its source and licence, signed with the author's key.
#[cfg(feature = "packs-publish")]
pub fn write(id: &str, face: Vec<u8>, source: &str, licence: &str, key: &[u8; 32]) -> Result<Vec<u8>, String> {
    use vsf::VsfType;
    let i = index(id).ok_or_else(|| format!("unknown font pack {id}"))?;
    let signer = ed25519_dalek::SigningKey::from_bytes(key).verifying_key().to_bytes().to_vec();
    let mut s = vsf::VsfSection::new("font");
    s.add_field("pack", VsfType::x(id.to_string()));
    s.add_field("family", VsfType::x(PACKS[i].family.to_string()));
    s.add_field("source", VsfType::x(source.to_string()));
    s.add_field("licence", VsfType::x(licence.to_string()));
    let n = face.len();
    s.add_field("face", VsfType::t_u3(vsf::types::Tensor::new(vec![n], face)));
    let doc = vsf::VsfBuilder::new().creation_time_oscillations(vsf::eagle_time_oscillations()).signed_only(VsfType::ke(signer)).add_section_direct(s).build().map_err(|e| format!("build: {e}"))?;
    vsf::verification::sign_file(doc, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_script_finds_its_pack() {
        assert_eq!(need('a'), None);
        assert_eq!(need('Ж'), None, "Cyrillic is in the binary's Open Sans");
        assert_eq!(need('ש'), index("noto-sans-hebrew").map(Need::Script));
        assert_eq!(need('ق'), index("noto-sans-arabic").map(Need::Script));
        assert_eq!(need('த'), index("noto-sans-tamil").map(Need::Script));
        assert_eq!(need('ລ'), index("noto-sans-lao").map(Need::Script));
        assert_eq!(need('ጠ'), index("noto-sans-ethiopic").map(Need::Script));
        assert_eq!(need('の'), Some(Need::Kana));
        assert_eq!(need('한'), Some(Need::Hangul));
        assert_eq!(need('京'), Some(Need::Shared));
        for p in PACKS {
            assert!(index(p.id).is_some());
        }
        for h in [Han::Simplified, Han::Traditional, Han::HongKong, Han::Japanese, Han::Korean] {
            assert!(h.pack() < PACKS.len());
        }
    }

    /// Text drawn without its face asks for the right pack once: Han alone the default region, kana Japanese, Hangul Korean, Hebrew its own; what the binary carries asks for nothing.
    #[test]
    fn text_asks_for_its_pack_once() {
        let mut t = crate::text::TextRenderer::new();
        let style = crate::text::TextStyle::new(16.0, 0xFFFF_FFFF);
        let asked = |t: &mut crate::text::TextRenderer, text: &str| {
            t.measure_text(text, &style);
            t.wanted_packs().iter().map(|p| p.id).collect::<Vec<_>>()
        };
        assert!(asked(&mut t, "Seattle, Москва, Αθήνα").is_empty());
        assert_eq!(asked(&mut t, "北京市"), ["noto-sans-sc"]);
        assert_eq!(asked(&mut t, "東京都 しぶや"), ["noto-sans-jp"]);
        assert_eq!(asked(&mut t, "서울"), ["noto-sans-kr"]);
        assert_eq!(asked(&mut t, "תל אביב"), ["noto-sans-hebrew"]);
        assert!(asked(&mut t, "北京市 תל אביב").is_empty(), "each is offered once");
        t.set_han(Han::Traditional);
        assert_eq!(asked(&mut t, "臺北市"), ["noto-sans-tc"]);
        t.offer_again("noto-sans-hebrew");
        assert_eq!(asked(&mut t, "ירושלים"), ["noto-sans-hebrew"], "offered again after a failed fetch");
        assert!(t.add_pack_face("noto-sans-hebrew", vec![0; 16]).is_err(), "bytes that are not the face");
    }

    #[cfg(feature = "packs-publish")]
    #[test]
    fn a_pack_reads_back_and_refuses_another_key() {
        let key = [7u8; 32];
        let author = ed25519_dalek::SigningKey::from_bytes(&key).verifying_key().to_bytes();
        let doc = write("noto-sans-hebrew", vec![1, 2, 3, 4], "test", "OFL", &key).unwrap();
        let (i, face) = read(&doc, author).unwrap();
        assert_eq!((PACKS[i].id, face), ("noto-sans-hebrew", vec![1, 2, 3, 4]));
        assert!(read(&doc, [9u8; 32]).is_err(), "another author's key");
        let mut bad = doc.clone();
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert!(read(&bad, author).is_err(), "a changed byte");
    }
}
