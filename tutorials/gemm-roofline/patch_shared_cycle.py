from pathlib import Path
path = Path("src/gpgpu-sim/shader.cc")
text = path.read_text()
old = '''    if (!inst.m_smem_exposed) {
      svc.expose(shared_memory_service_t::LSU, bytes);
      inst.m_smem_exposed = true;
      stall = true;
'''
new = '''    if (!inst.m_smem_exposed) {
      // Count each shared-memory instruction once when it first enters the
      // shared-memory service. Later service waits, bank conflicts, and
      // pipeline backpressure must not add traffic again.
      roofline_stats::add_l1(inst.is_store(), bytes);
      svc.expose(shared_memory_service_t::LSU, bytes);
      inst.m_smem_exposed = true;
      stall = true;
'''
old_tail = '''  } else {
    // Hierarchical Roofline L1 represents the complete L1TEX/shared-memory
    // level. Charge each completed shared-memory instruction exactly once;
    // stalled bank-conflict retries are not counted again.
    const unsigned bytes =
        inst.active_count() * (inst.data_size ? inst.data_size : 4u);
    roofline_stats::add_l1(inst.is_store(), bytes ? bytes : 4u);
    rc_fail = NO_RC_FAIL;
  }
'''
new_tail = '''  } else {
    rc_fail = NO_RC_FAIL;
  }
'''
if text.count(old) != 1:
    raise SystemExit(f"first-submit block matches: {text.count(old)}")
if text.count(old_tail) != 1:
    raise SystemExit(f"completion block matches: {text.count(old_tail)}")
path.write_text(text.replace(old, new, 1).replace(old_tail, new_tail, 1))
print("patched shared-memory traffic counting")
