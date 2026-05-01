# AGENT.md - Code Generation Rules
## Rule 0: Bounds Checks and Saturating Arithmetic
**IF YOU ADD ANY BOUNDS CHECK OR SATURATING ARITHMETIC, YOU ARE REQUIRED TO:**
0. **STATE WHY** it was added
1. **PROVE** it was necessary
2. **EXPLAIN** what undefined behavior or memory unsafety it prevents

### Complete List of Operations That Require Justification:

**Saturating Arithmetic:**
- `saturating_add`, `saturating_sub`, `saturating_mul`, `saturating_div`, `saturating_pow`
- `wrapping_add`, `wrapping_sub`, `wrapping_mul` (when used to prevent overflow)
- `checked_add`, `checked_sub`, `checked_mul`, `checked_div`, `checked_rem`, `checked_pow`
- `overflowing_*` operations

**Clamping/Range Limiting:**
- `clamp(min, max)`
- `min(a, b)`, `max(a, b)` when used to constrain values
- Manual clamping: `if x > max { max } else { x }`

**Bounds Checks:**
- `if idx < vec.len()` or any length/capacity check
- `.get(idx)` instead of `[idx]`
- `.get_mut(idx)` instead of `[idx]`
- Range checks before indexing: `if x < width && y < height`

**Safe Access Patterns:**
- `.get().unwrap_or()`, `.get().unwrap_or_default()`
- `.get().map()`, `.and_then()`
- Pattern matching on `.get()` results

**Division/Modulo Guards:**
- `if divisor != 0` before division
- `checked_div`, `checked_rem`

**Bit Shift Validation:**
- Checking shift amount < type bit width
- `checked_shl`, `checked_shr`

### What Counts as a Bounds Check:
- `if idx < vec.len()`
- `vec.get(idx)` instead of `vec[idx]`
- `.checked_add()`, `.saturating_add()`, `.saturating_subtract()` (unless explicitly requested)
- Any conditional that guards array/slice access

### When Bounds Checks ARE Required:
```rust
// When final_x and final_y are negative, when cast to usize they fail the width/height check
if (final_x as usize) < width && (final_y as usize) < height {
    let idx = final_y as usize * width as usize + final_x as usize;
    pixels[idx] = blended as u32;  // Would panic without the check
}
```
**WHY**: `final_x` and `final_y` come from glyph positioning math and can be negative or exceed buffer dimensions. The cast to `usize` makes negatives wrap to huge values (fail bigger than shit), and the check prevents out-of-bounds memory access.

**PROOF**: Without this check, we'd attempt to write outside the pixel buffer and panic.

### When Bounds Checks Are FORBIDDEN:
```rust
// Internal loop with known bounds - NO CHECKS
for y in 0..self.height as usize {
    for x in 0..self.width as usize {
        let idx = y * self.width as usize + x;
        pixels[idx] = color;  // idx is MATHEMATICALLY proven in-bounds
    }
}
```
```rust
// Packed SIMD arithmetic - NO SATURATING OPS
let mut blended = bg * inv_alpha + colour * alpha;  // Overflow is part of the algorithm
blended = (blended >> 8) & 0x00FF00FF00FF00FF;     // Bit ops handle the overflow correctly
```

### When to Panic Instead:
```rust
// Debug visualization - if buffer is wrong size, FAIL LOUD, that means WE FUCKED UP SOMEWHERE ELSE, bounds checks hide problems!
if self.show_textbox_mask {
    for y in 0..self.height as usize {
        for x in 0..self.width as usize {
            let idx = y * self.width as usize + x;
            let alpha = self.textbox_mask[idx];  // PANIC if uninitialized = correct behavior
            pixels[idx] = pack_argb(alpha, alpha, alpha, 255);
        }
    }
}
```

## Core Principles

### 0. **Trust the Math**
- If loop bounds guarantee safety, don't add runtime checks
- Index calculations from known dimensions are proof of correctness
- Compiler optimization depends on removing redundant bounds checks

### 1. **Fail Fast, Fail Loud**
- Panics are better than silent corruption
- A panic means "there's a bug in initialization" not "add a bounds check"
- Debug visualizations should expose bugs, not hide them

### 2. **Understand Packed SIMD**
- Overflow in packed arithmetic is often **required** for correctness
- Bit masks and shifts handle channel isolation and reconstruction
- Saturating ops break the mathematical properties being exploited

### 3. **Respect Explicit Unsafe Contracts**
```rust
// SAFETY: idx is proven in-bounds by loop invariant (0..width * 0..height)
unsafe { pixels.get_unchecked_mut(idx) }
```
If there's a SAFETY comment, **read it**. It's there because the human proved correctness.
HashMap shall NOT be used without explicit consent and proof showing how it is faster/better than a linear search or a simple lookup.

## The Clamp Trap

**`clamp()` is defensive programming that hides bugs.**

```rust
// WASTEFUL - clamp does nothing useful here
let byte_value = pixel_value.clamp(0.0, 255.0) as u8;

// CORRECT - cast already handles bounds
let byte_value = pixel_value as u8;
```

**Why clamp is probably wrong:**
0. **Casting already handles bounds** - `f32 as u8` truncates automatically, the clamp checks bounds the cast does for free
1. **Hides bugs** - if values are outside range, you WANT to know (forensic), not silently fix it (defensive)
2. **Assumes your math is broken** - if calculations are correct, values should never be out of range anyway

**The forensic approach:** If the cast wraps/truncates unexpectedly, that exposes the real bug in your math. Fix the math, don't hide the symptoms.

## When Unsure

**ASK.** Don't add "defensive" checks. If you're not sure whether a bounds check is needed:
0. State what you're unsure about
1. Show the code without the check
2. Explain what would happen if the invariant is violated
3. Let the human decide

## When Needed

**PROOF REQUIRED** Don't add clamping, min, max, saturating or any other clamping ops unless it has been proven necessary AND accepted by the user!

## Error Handling Philosophy

0. **Initialization bugs should panic** - if textbox_mask is empty, that's a bug in `new()`, not in the render loop
1. **External input should be validated** - user input, file data, network packets get bounds checks
2. **Internal invariants should be maintained** - if your loop guarantees safety, assert it in debug builds if needed, but don't check in release

## Dimensional Units

**NEVER use fixed pixel values.** 20px on an 8K TV ≠ 20px on a watch. Use relative scaling:

```rust
// WRONG - fixed pixels break across displays
let margin_pixels = 20;

// CORRECT - scale to display dimensions
let margin = box_width / 40;  // 2.5% of textbox width
let margin = self.span / 64;  // Fraction of span
```

**Physics scales, pixels don't.** E=mc² works everywhere because it's based on fundamental relationships, not arbitrary units. Your code should too.

### Universal Scaling Units (Already Available):
- `self.span` - harmonic mean of width and height = 2wh/(w+h), universal scaling base
- `self.perimeter` - width + height, for edge-aware calculations
- `self.diagonal_sq` - width² + height², for distance calculations

### Why Span Uses Harmonic Mean

The harmonic mean `2wh/(w+h)` has unique properties that make it ideal for UI scaling:

0. **Smooth at w==h** - No discontinuity when aspect ratio crosses 1:1 (unlike min/max)
1. **Finite slope at axes** - Behaves well as either dimension approaches zero
2. **Slope exactly 1 along diagonal** - Natural scaling along the w==h line
3. **Biased toward smaller dimension** - UI elements scale appropriately on narrow displays

Compare alternatives:
- `min(w,h)` - Discontinuous derivative at w==h, creates visual "jumps"
- `max(w,h)` - Same discontinuity problem
- `sqrt(w*h)` (geometric mean) - Smooth, but infinite slope at axes
- `(w+h)/2` (arithmetic mean) - Doesn't bias toward smaller dimension

The harmonic mean is the unique function with all desired properties.

Use these. Derive everything from screen dimensions and their mathematical relationships.

## Decimal Indexing is FORBIDDEN

**NEVER use decimal digits (0-9) for array indices, field names, or any programmatic counting.**

### Why Decimal Indexing is Broken:

0. **CPUs count in binary** - every decimal index requires base conversion overhead
1. **Off-by-one bugs** - humans start at 1, computers at 0, decimal encourages confusion
2. **Fragile serialization** - string concatenation like `s{idx}_` creates parse nightmares
3. **Not how hardware works** - you're teaching incorrect computational models

### The Right Way: VSF Nested Structures

```rust
// WRONG - decimal string prefixes, flat namespace pollution
for idx in 0..party_slots.len() {
    let prefix = format!("s{}_", idx);  // s0_, s1_, s2_... DECIMAL GARBAGE
    fields.push((format!("{}handle_hash", prefix), ...));
    fields.push((format!("{}offer_x25519_pub", prefix), ...));
}
```

```rust
// CORRECT - proper nested VSF sections
let mut slots_section = VsfSection::new("party_slots");
for slot in &party_slots {
    let mut slot_section = VsfSection::new("slot");
    slot_section.add_field("handle_hash", VsfType::hb(slot.handle_hash));
    slot_section.add_field("offer_x25519_pub", VsfType::kx(slot.offer.x25519_pub));
    slots_section.add_nested_section(slot_section);
}
```

**If you find yourself doing string formatting with decimal indices, STOP. You're doing it wrong. Use VSF's native array/section nesting.**

## VSF Type Markers Are Self-Describing

**NEVER rely on position to determine what a value is. The type marker tells you.**

VSF key types are self-describing:
- `kx` = X25519 public key (32B)
- `kf` = FrodoKEM public key (~15KB)
- `kn` = NTRU public key (~1KB)
- `kl` = Classic McEliece public key (~524KB)
- `kh` = HQC public key (~7KB)
- `kk` = secp256k1 public key (33B)
- `kp` = P-curve key (disambiguate by size: 97B = P-384, 65B = P-256)
- `ks` = Shared secret (any KEM output)

### WRONG: Positional Parsing

```rust
// FRAGILE - what if order changes? Silent corruption!
slot.offer = Some(ClutchFullOfferPayload {
    x25519_public: extract_key(&values[idx])?,     // Assumes position 0 is x25519
    frodo_public: extract_key(&values[idx + 4])?,  // Assumes position 4 is frodo
});
```

### CORRECT: Match on Type Marker

```rust
// ROBUST - type marker tells us exactly what it is
for v in &values {
    match v {
        VsfType::kx(b) => offer.x25519_public = b.try_into()?,
        VsfType::kf(b) => offer.frodo_public = b.clone(),
        VsfType::kn(b) => offer.ntru_public = b.clone(),
        VsfType::kl(b) => offer.mceliece_public = b.clone(),
        VsfType::kh(b) => offer.hqc_public = b.clone(),
        VsfType::kk(b) => offer.secp256k1_public = b.clone(),
        VsfType::kp(b) if b.len() == 97 => offer.p384_public = b.clone(),
        VsfType::kp(b) if b.len() == 65 => offer.p256_public = b.clone(),
        _ => {}
    }
}
```

**Why**: If you parse `kl` (McEliece) into a field expecting `kf` (Frodo), you get silent corruption.
The type marker exists precisely so you never have to guess. Use it.

## Protocol Evolution: No Fork Bullshit

**We control all clients. All 5 of them are on your desk. Break the protocol, update everything, move on.**

### FORBIDDEN: Backwards Compatibility Theater
- No "v1" vs "v2" protocol forks
- No feature flags for legacy clients
- No "if version < X then do broken thing" code paths

### REQUIRED: Atomic Updates
- Protocol change? Update all clients simultaneously
- Old clients that can't parse new format should **fail loudly** with version mismatch
- VSF version fields exist for forensics, not branching logic

**Why**: Backwards compatibility is how good protocols become IEEE-754. You have total control. Use it.

## Language Preferences

### Strongly Preferred:
- **Rust**: Memory safety through ownership, zero-cost abstractions
- **Assembly**: When you need exact control
- **Metal**: GPU compute with known performance characteristics

### Not Allowed:
- **Python**: Slow, loose typing causes bugs, cannot copy-paste, 1-indexed nonsense infected everything
- High-level scripting when systems programming is needed, text parsing, terribly unsafe

## VSF Serialization: Use High-Level APIs

**ALWAYS prefer VSF's schema-validated builders over manual byte manipulation.**

### The Right Way: SectionBuilder + SectionSchema

```rust
use vsf::schema::{SectionSchema, SectionBuilder, TypeConstraint};

// Define schema (or use official: vsf::schema::official::network_peer_schema())
let schema = SectionSchema::new("announce")
    .field("challenge_hash", TypeConstraint::Blake3Rolling)
    .field("handle_hash", TypeConstraint::Blake3Provenance)
    .field("port", TypeConstraint::AnyUnsigned);

// Build with validation
let bytes = schema.build()
    .set("challenge_hash", VsfType::hb(hash))?
    .set("handle_hash", VsfType::hb(handle_hash))?
    .set("port", 41641u16)?
    .encode()?;

// Parse → modify → re-encode
let mut builder = SectionBuilder::parse(schema, &bytes)?;
builder = builder.set("port", 8080u16)?;
let updated = builder.encode()?;
```

### FORBIDDEN: Manual Serialization

```rust
// NO - manual byte pushing, error-prone, no validation
let mut bytes = Vec::new();
bytes.push(b'[');
bytes.extend(VsfType::d("announce".to_string()).flatten());
bytes.push(b'(');
bytes.extend(VsfType::d("port".to_string()).flatten());
bytes.push(b':');
bytes.extend(VsfType::u4(41641).flatten()); // This becomes fragile. If port is less than 255 a u3 is valid.
bytes.push(b')');
bytes.push(b']');
```

### Why High-Level APIs:

0. **Type safety** - TypeConstraint validates values match expected types
1. **Schema validation** - Unknown fields are caught, required fields enforced
2. **Round-trip safe** - parse → modify → encode workflow guaranteed correct
3. **Self-documenting** - Schema IS the documentation
4. **Future-proof** - Wire format changes handled by library, not your code

### Available Official Schemas:

- `vsf::schema::official::image_schema()` - Image metadata
- `vsf::schema::official::camera_schema()` - Camera hardware config
- `vsf::schema::official::audio_schema()` - Audio stream metadata
- `vsf::schema::official::network_peer_schema()` - P2P peer info
- `vsf::schema::official::announce_schema()` - FGTW bootstrap

### For Complete Files: VsfBuilder

```rust
use vsf::vsf_builder::VsfBuilder;

let bytes = VsfBuilder::new()
    .add_section(my_section)
    .provenance(provenance_hash)  // Immutable identity
    .build()?;  // BLAKE3 hash computed automatically
```

**The library handles integrity hashing, header layout, section offsets - you just add content.**

## VSF Transport Rule: COMPLETE FILES ONLY

**ALL network transport and disk storage MUST use complete VSF files.**

A complete VSF file has (bare minimum):
- `RÅ<` magic header
- Version (`z`) and backward compat (`y`)
- Header length (`b`)
- Creation timestamp (`ef5` or `ef6`)
- Provenance hash (`hp`) - REQUIRED for integrity
- Field count (`n`)
- Header end marker (`>`)
- Section structure with `[ ]` delimiters

### FORBIDDEN: Raw VSF Types

```rust
// NO - raw VSF type without file wrapper (no integrity, no timestamp, no version)
let bytes = VsfType::v(b'e', encrypted_data).flatten();
Response::from_bytes(bytes)

// NO - bare section without file header
let bytes = section.encode();
socket.send(&bytes);
```

### REQUIRED: Complete VSF Files

```rust
// YES - complete VSF file with proper header and provenance hash
let bytes = VsfBuilder::new()
    .creation_time_nanos(timestamp)
    .provenance_only()
    .add_section("encrypted_data", vec![
        ("payload".to_string(), VsfType::v(b'e', encrypted_data)),
    ])
    .build()?;
```

**Why**: Raw types have no integrity protection, no timestamps, no version info.
The provenance hash ensures data hasn't been tampered with in transit.
Every VSF on the wire or disk can be inspected with `vsfinfo` and verified.

## Code Style

### Terminal Commands:
```bash
# DON'T add comments in commands
cargo build --release

# DO separate commands that should be run individually
cargo clean
cargo build --release
```

### Prefer Explicit Over "Safe":
```rust
// YES - explicit, clear, no overhead
pixels[idx] = color;

// NO - hiding potential bugs, runtime overhead
if let Some(pixel) = pixels.get_mut(idx) {
    *pixel = color;
}
```

## When Unsure

**ASK.** Don't add "defensive" checks. If you're not sure whether a bounds check is needed:
0. State what you're unsure about
1. Show the code without the check
2. Explain what would happen if the invariant is violated
3. Let the human decide

## Persistence Rule: EVERY CHANGE HITS DISK

**Any state change beyond a single keystroke (and ping/pong) MUST be persisted immediately.**

### What MUST be Saved:
- Messages sent or received → `save_messages()` immediately
- Contact state changes (online status, CLUTCH completion) → `save_contact_state()`
- Friendship chain state → `save_friendship()` after every advance
- User preferences/settings → save on change, not on exit

### Why:
- App can crash at any moment (kernel panic, power loss, OOM killer)
- RAM is ephemeral. Disk is truth.
- If you can't see it in `~/.config/photon/`, it didn't happen

### The Pattern:
```rust
// Message received - save IMMEDIATELY after adding to list
contact.messages.push(ChatMessage::new(text, false));
save_messages(&contact, &our_seed, &device_secret)?;  // RIGHT HERE

// NOT "batch save on exit" - that guarantees data loss
```

**Think like a database:** every write is a commit. There is no rollback. There is no "save later."

## Time Standards: Eagle Time ONLY

**UNIX time, UTC timestamps, ISO 8601, and ALL ambiguous time references are FORBIDDEN.**

### The ONLY Acceptable Time Standard: Eagle Time

Eagle Time is defined as:
- **Epoch**: July 20, 1969, 20:17:40 UTC (Apollo 11 lunar landing - "The Eagle has landed")
- **Unit**: Eagle seconds = 1,420,407,826 hydrogen-1 hyperfine transition periods (21cm line)
- **Same duration as SI seconds** - only the epoch differs
- **Reference frame**: Milky Way-Andromeda barycentric frame (accounts for gravitational time dilation)

### Why Eagle Time Exists

UNIX time is fundamentally broken:
- **Epoch meaningless**: January 1, 1970 - arbitrary bureaucratic date with no physical significance
- **Leap second insanity**: Breaks monotonicity, causes distributed system failures, fundamentally ambiguous
- **No physical definition**: "SI seconds since 1970" - but measured where? GPS satellites? Earth's surface? What gravitational potential?
- **Zone confusion**: "UTC" hides timezone complexity, leads to timestamp ambiguity

Eagle Time solves ALL of these:
- **Physical epoch**: Apollo 11 landing - unambiguous historical moment, commemorates humanity's greatest engineering achievement
- **Physically defined second**: Hydrogen-1 hyperfine transition - measurable by any civilization with a 21cm radio receiver
- **Relativistic clarity**: Explicitly measured at Milky Way-Andromeda barycentric frame
- **No leap seconds**: Eagle Time is monotonic, no discontinuities, no special cases

### REQUIRED: All Time Values Use Eagle Time

```rust
// CORRECT - Eagle Time with explicit type
use vsf::eagle_time::{EagleTime, EtType, datetime_to_eagle_time, eagle_time_nanos};

let timestamp = EagleTime::new(EtType::f6(eagle_time_nanos()));
let creation_time = datetime_to_eagle_time(chrono::Utc::now());

// Store in VSF with 'e' type marker
VsfType::e(timestamp)
```

### FORBIDDEN: UNIX Time and Ambiguous Timestamps

```rust
// NO - UNIX timestamps have no place in this codebase
let unix_time = SystemTime::now().duration_since(UNIX_EPOCH)?;  // WRONG EPOCH

// NO - "UTC" without Eagle Time conversion
let utc_now = Utc::now();  // Ambiguous - no conversion to Eagle Time

// NO - Raw chrono::DateTime without Eagle Time wrapper
fields.push(("timestamp", VsfType::u6(utc.timestamp() as u64)));  // UNIX time smuggled in

// NO - ISO 8601 strings (ambiguous, text-based, bloated)
let iso = "2025-12-19T15:30:00Z";  // Use EagleTime::to_datetime() for display ONLY
```

### Allowed Conversions: Display Only

You MAY convert Eagle Time to human-readable formats for **display purposes only**:

```rust
// ALLOWED - Eagle Time for storage, chrono for human display
let eagle_timestamp = EagleTime::new(EtType::f6(eagle_time_nanos()));
let display_time = eagle_timestamp.to_datetime();  // Convert to UTC DateTime for formatting
println!("Message sent: {}", display_time.format("%Y-%m-%d %H:%M:%S"));

// BUT - NEVER store the chrono::DateTime, only display it
// Storage MUST use Eagle Time
```

### When You're Tempted to Use UNIX Time

**STOP. You're doing it wrong.**

If you find yourself writing:
- `UNIX_EPOCH`
- `.timestamp()` on a `DateTime`
- `SystemTime::now().duration_since(...)`
- Storing raw `u64` seconds without Eagle Time wrapper
- Any date/time calculation that doesn't go through `eagle_time.rs`

**You are violating this rule.** Convert to Eagle Time immediately.

### The Eagle Time Flow

```
External source → chrono::DateTime → datetime_to_eagle_time() → EagleTime → Store in VSF
                                                                          ↓
VSF storage ← EagleTime ← Parse from VSF ← Network/Disk ← Serialized VSF with 'e' type
     ↓
EagleTime → to_datetime() → chrono::DateTime → Display formatting → Human-readable string
```

**NEVER skip the Eagle Time conversion.** Every timestamp in VSF MUST be Eagle Time.

### Why This Matters

When you use UNIX time, you're teaching:
- Arbitrary epochs are acceptable (they're not - physics matters)
- Leap seconds are a solved problem (they're not - they break systems)
- Timestamps without physical definitions are fine (they're not - ambiguity kills)

Eagle Time is the physically correct, unambiguous, monotonic time standard. Use it.

## Active Projects

- **Spirix**: Two's complement floating-point (replaces IEEE-754)
- **TOKEN**: Cryptographic identity/reputation system
- **Ferros**: Killswitch-ready Rust OS
- **VSF**: Unlimited serialization format

---

*Remember: The user is rebuilding the computational stack from first principles. They see patterns others miss. They're right that base-10 is broken and IEEE-754 teaches incorrect math. When they write packed SIMD, they know exactly what they're doing. Your job is to help build the cathedral, not childproof it.*

Oh, and if you haven't noticed by now but when indexing in //'s and variable names, starte with the first number (0), not the second number (1)
Development builds? use ./build-development.sh DO NOT DO RELEASE BUILDS UNLESS THE USER EXPLICITLY ASKS!
Agent to test builds with ./build-development.sh and user will run.

## Comment Formatting

**Don't wrap doc-comment paragraphs across multiple `///` or `//!` lines.** One paragraph = one continuous line, however long it needs to be. Bullet lists wrap naturally per bullet, but each bullet's text stays on one line. No `rustfmt.toml` lives in this project tree by design — manual line wraps cost vertical space without benefit.

```rust
// WRONG — one paragraph artificially broken into three lines, wastes vertical space
/// Sets the RU multiplier. The RU multiplier controls how many pixels
/// correspond to 1 RU. The default is 1.0; consumers scale this to match
/// their UI density (e.g. set to 1/64 for em-like sizing).

// CORRECT — one paragraph, one line, however long it needs to be
/// Sets the RU multiplier. The RU multiplier controls how many pixels correspond to 1 RU. The default is 1.0; consumers scale this to match their UI density (e.g. set to 1/64 for em-like sizing).
```

Same principle for code: don't pre-emptively split function signatures or expressions to fit 80 chars; only break when it genuinely improves readability. Don't run `cargo fmt` with default settings — leave manual line lengths alone.

## Persistence Cadence for Streaming UI Events

**Refines `## Persistence Rule` above.** That rule says every change hits disk; this clarifies what "a change" means for high-frequency UI events.

Drag, resize, text-entry typing, scroll, and similar 60fps-rate state mutations must **not** hit disk per-frame. A 60fps drag would write 60 layout VSF files per second — pointless wear and IO load. Coalesce them.

- **Max write frequency during a streaming gesture: 1 Hz**, and ideally lower. A long-running drag should not produce a torrent of disk writes.
- **On gesture release** (mouse-up, key release, focus-loss, drag-end), flush the pending write immediately so the final state lands on disk.
- **Explicit user actions** (save command, preference toggle, setting change, "OK" button) bypass the debounce — persist immediately, exactly per the rule above.

The streaming case is `every-change-hits-disk` where the coalesced write **is** the change worth persisting. The user-facing intent is "the layout I left it in is what I'll see next launch," not "every intermediate frame of a drag is durable."

## Coord Storage: f32 Primary, Spirix Optional

**Default storage type for window dimensions, pane positions, and UI canvas coords is `f32`.** Hardware has native f32 add/mul on every relevant target (NEON `fadd`/`fmul`, AVX/SSE `addps`/`mulps`); Spirix is software-emulated. Photon and other consumers run as fast as their pre-fluor code only if the layout layer is f32.

**Spirix support is welcome where it specifically matters** — precision-critical rasterizer paths, deterministic-zoom apps (mandelbrot), or ferros builds that opt in via a future `spirix-coord` feature flag. Don't make Spirix the default just because it's elegant.

The goal: an f32 subpixel-aware center-oriented canvas, replacing photon's current integer/float mismatch with one consistent type throughout the layout layer.

## API / Implementation Separation: Hot-Swappable Internals

**The public API is stable and simple. The internals are whatever runs best, and they must be hot-swappable.** Consumers give fluor center-based RU coordinates in `f32` and a render target; whatever fluor does under the hood — CPU scalar blit, NEON SIMD, Spirix-precision rasterizer for AA, GPU offload, runtime backend switch — is invisible.

This applies broadly:
- **Renderer backends** (CPU scalar / CPU SIMD / Spirix-precision / GPU): default to enum dispatch for runtime swap, feature-flag for fixed builds when a consumer wants only one path compiled in. Both are valid — pick per-component based on perf vs flexibility.
- **Blit kernels** (NEON / SSE2 / AVX2 / scalar): runtime feature-detected, dispatched once per frame or once per blit, invisible.
- **Coord precision in non-API paths**: an internal Spirix subpath for squircle AA or extreme-zoom content is fine and invisible. The API surface is still `RuVec2 { x: f32, y: f32 }`.

This is **not** in conflict with `Protocol Evolution: No Fork Bullshit` — that rule is about external wire format and protocol stability, where backwards-compat layers rot. Internal renderer hot-swap is implementation detail; the API stays exactly one shape.