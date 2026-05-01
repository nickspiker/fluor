#!/bin/bash
set -e

echo "Building fluor (development profile)..."
cargo build

echo ""
echo "Running tests..."
cargo test
