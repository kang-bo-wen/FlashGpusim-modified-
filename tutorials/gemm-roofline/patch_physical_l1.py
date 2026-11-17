from pathlib import Path

h = Path('src/abstract_hardware_model.h')
s = h.read_text()
s = s.replace('''    m_smem_exposed = false;
    m_smem_got = 0;
    m_current_lane = 0;''', '''    m_smem_exposed = false;
    m_smem_got = 0;
    m_smem_physical_bytes = 0;
    m_current_lane = 0;''')
s = s.replace('''  bool m_smem_exposed;
  unsigned m_smem_got;''', '''  bool m_smem_exposed;
  unsigned m_smem_got;
  unsigned m_smem_physical_bytes;''')
h.write_text(s)

cc = Path('src/abstract_hardware_model.cc')
s = cc.read_text()
old = '''      unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
      unsigned total_accesses = 0;
      for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;'''
new = '''      unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
      unsigned total_accesses = 0;
      unsigned physical_word_transactions = 0;
      for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''        if (m_config->shmem_limited_broadcast) {
          // step 2: look for and select a broadcast bank/word if one occurs'''
new = '''        // Each unique bank/word pair is one physical 32-bit shared-memory
        // transaction. Multiple lanes reading the same word are a broadcast
        // and consume one transaction; bank conflicts serialize transactions
        // but do not change their physical byte count.
        for (const auto &bank : bank_accs)
          physical_word_transactions += bank.second.size();

        if (m_config->shmem_limited_broadcast) {
          // step 2: look for and select a broadcast bank/word if one occurs'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''      cycles = total_accesses;  // shared memory conflicts modeled as larger
                                // initiation interval'''
new = '''      m_smem_physical_bytes = physical_word_transactions * m_config->WORD_SIZE;
      cycles = total_accesses;  // shared memory conflicts modeled as larger
                                // initiation interval'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
cc.write_text(s)

shader = Path('src/gpgpu-sim/shader.cc')
s = shader.read_text()
old = '''      // Count each shared-memory instruction once when it first enters the
      // shared-memory service. Later service waits, bank conflicts, and
      // pipeline backpressure must not add traffic again.
      roofline_stats::add_l1(inst.is_store(), bytes);'''
new = '''      // Count physical shared-memory traffic once: unique bank/word
      // transactions computed during memory-access generation.
      roofline_stats::add_l1(inst.is_store(), inst.m_smem_physical_bytes);'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
shader.write_text(s)
print('patched physical shared-memory traffic')
