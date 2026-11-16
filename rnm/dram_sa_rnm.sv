// Real-number model (RNM) of the DRAM latch sense amplifier.
//
// Behaviour captured from SPICE (see scripts/m5_rnm.py):
//   * decision    : sign of (v_bl - v_blb - VOS)
//   * resolve time: t = T0 + TAU * ln(VSW / |dv|), clamped below by TMIN
//                   (latch regeneration: the split grows as e^(t/TAU))
//   * metastable  : |dv| < DV_META -> output stays X and a warning is raised
//
// Time unit is ns (`timescale 1ns/1fs in the testbench). Voltages are volts.
`ifndef DRAM_SA_RNM_SV
`define DRAM_SA_RNM_SV
`timescale 1ns/1fs

module dram_sa_rnm #(
  parameter real T0      = 0.30,   // ns, fitted
  parameter real TAU     = 0.20,   // ns, regeneration time constant (fitted)
  parameter real VSW     = 1.44,   // V, split counted as "resolved" (80% of VDD)
  parameter real TMIN    = 0.80,   // ns, fastest resolution (enable-ramp limited)
  parameter real VOS     = 0.0,    // V, input-referred offset of this instance
  parameter real DV_META = 50e-6   // V, below this the latch is treated as metastable
) (
  input  real  v_bl,     // bitline voltage when the sense amp is enabled
  input  real  v_blb,    // reference bitline voltage
  input  real  v_os,     // extra offset (V), e.g. a Monte Carlo sample; tie to 0.0 if unused
  input  logic sae,      // sense enable (rising edge = SAN starts to fall)
  output logic dout,     // 1: BL resolved high, 0: BL resolved low, X: not resolved
  output real  t_res     // resolve time of the last decision (ns), for checking
);
  real dv, adv;

  initial dout = 1'bx;

  always @(posedge sae) begin : sense
    dv  = v_bl - v_blb - VOS - v_os;
    adv = (dv < 0.0) ? -dv : dv;
    dout = 1'bx;
    if (adv < DV_META) begin
      t_res = -1.0;
      $display("%t dram_sa_rnm: metastable input |dv| = %g V", $realtime, adv);
    end else begin
      t_res = T0 + TAU * $ln(VSW / adv);
      if (t_res < TMIN) t_res = TMIN;
      #(t_res) dout = (dv > 0.0);
    end
  end

  // precharge: releasing the enable clears the latch
  always @(negedge sae) begin
    disable sense;
    dout = 1'bx;
  end
endmodule
`endif
