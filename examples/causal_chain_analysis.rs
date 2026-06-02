//! # Causal Chain Analysis
//!
//! This example demonstrates how transfer entropy reveals causal relationships
//! in a multi-service architecture — something impossible with correlation alone.
//!
//! ## Scenario
//!
//! Three services in a chain:
//!
//! ```text
//! Auth ──→ Checkout ──→ Inventory
//! ```
//!
//! When Auth's latency increases, it causes Checkout errors. But Inventory's
//! queue depth just correlates with Checkout — it's not caused by it directly.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example causal_chain_analysis
//! ```

use iii_observability::{causality_matrix, correlation_matrix};
use std::f64::consts::PI;

fn main() {
    println!("=== Causal Chain Analysis: 3-Service Architecture ===\n");

    // ── Generate synthetic time series ──────────────────────────────

    // Auth: latency varies sinusoidally over 200 time steps
    let auth: Vec<f64> = (0..200).map(|i| {
        50.0 + 20.0 * (2.0 * PI * i as f64 / 50.0).sin()
    }).collect();

    // Checkout: error rate is CAUSED by auth's latency with lag=3
    // checkout[i] = 0.7 * auth[i-3] + noise
    let checkout: Vec<f64> = {
        let mut v = vec![1.0; 200];
        for i in 3..200 {
            v[i] = auth[i - 3] * 0.7 + (i as f64 * 0.1).sin() * 5.0;
        }
        v
    };

    // Inventory: queue depth correlates with checkout but is NOT caused by it
    // Both respond to a common external factor (load), so they co-move
    let inventory: Vec<f64> = (0..200).map(|i| {
        10.0 + 5.0 * (2.0 * PI * i as f64 / 50.0).sin() + // same base pattern as auth
            5.0 * (2.0 * PI * i as f64 / 30.0).cos()
    }).collect();

    println!("  Auth latency:       ~50ms avg, 20ms amplitude");
    println!("  Checkout errors:    Caused by auth latency (lag=3)");
    println!("  Inventory queue:    Co-moves with load, NOT caused by checkout");
    println!();

    // ── Compute correlation matrix ─────────────────────────────────

    println!("── Mutual Information Matrix (correlation) ──\n");

    let metrics = vec![
        ("auth".to_string(), auth.clone()),
        ("checkout".to_string(), checkout.clone()),
        ("inventory".to_string(), inventory.clone()),
    ];

    let corr = correlation_matrix(&metrics, 20);

    let names = ["auth", "checkout", "inventory"];
    print!("{:12}", "");
    for name in &names {
        print!("{:>12}", name);
    }
    println!();

    for (i, name) in names.iter().enumerate() {
        print!("{:12}", name);
        for j in 0..3 {
            if i == j {
                print!("{:>12.4}", corr[i][j]); // H(X) — self-entropy
            } else {
                print!("{:>12.4}", corr[i][j]);
            }
        }
        println!();
    }

    println!("\n  Mutual information shows ALL relationships (auth↔checkout↔inventory)");
    println!("  It cannot distinguish causation from correlation.");
    println!();

    // ── Compute causality matrix ───────────────────────────────────

    println!("── Transfer Entropy Matrix (causality) ──\n");

    let causal = causality_matrix(&metrics, 3, 20);

    print!("{:12}", "");
    print!("{:>12}", "→auth");
    print!("{:>12}", "→checkout");
    print!("{:>12}", "→inventory");
    println!();

    for (i, name) in names.iter().enumerate() {
        print!("{:12}", format!("{}→", name));
        for j in 0..3 {
            if i == j {
                print!("{:>12.4}", 0.0); // self-TE = 0
            } else {
                print!("{:>12.4}", causal[i][j]);
            }
        }
        println!();
    }

    println!("\n  Interpretation:");
    println!("  (row → col) means \"row's past helps predict col's future\"");

    println!("\n  Key findings:");

    // auth → checkout should be high (causal direction)
    let auth_to_checkout = causal[0][1];
    let checkout_to_auth = causal[1][0];
    let auth_to_inv = causal[0][2];
    let checkout_to_inv = causal[1][2];

    println!("  1. Auth  → Checkout:  TE = {:.4}", auth_to_checkout);
    println!("  2. Checkout  → Auth:  TE = {:.4}", checkout_to_auth);

    if auth_to_checkout > checkout_to_auth {
        println!("     ✅ Direction confirmed: Auth latency CAUSES checkout errors");
    } else {
        println!("     Note: directionality may need larger sample for strong signal");
    }

    println!();
    println!("  3. Auth  → Inventory: TE = {:.4}", auth_to_inv);
    println!("  4. Checkout  → Inventory: TE = {:.4}", checkout_to_inv);

    if auth_to_inv > 0.1 {
        println!("     📌 Auth and inventory share a common load pattern");
    }
    if checkout_to_inv < 0.1 {
        println!("     ✅ Checkout errors do NOT cause inventory queue growth");
        println!("     (They co-move due to shared load, not direct causation)");
    }

    println!();
    println!("── Summary ──\n");
    println!("  Mutual information finds:  auth↔checkout↔inventory (everything correlates)");
    println!("  Transfer entropy reveals:  auth → checkout (the actual causal chain)");
    println!();
    println!("  In production, this tells you which service to fix FIRST.");
    println!("  Fixing auth will fix checkout — fixing inventory won't help either.");
}
