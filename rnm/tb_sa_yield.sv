// Statistical read test with the RNM: every read draws a fresh sense-amp offset
// v_os ~ N(0, SA_SIGMA_OS) (Monte Carlo sigma from SPICE) and reads a stored '1'
// whose bitline signal is DV. Counts wrong or unresolved decisions per DV.
// 20,000 reads per point run in seconds; the same statistics in SPICE would take hours.
//   out: rnm/rnm_yield.csv (dv_V, n_reads, n_fail)
`timescale 1ns/1fs
`include "sa_params.svh"

module tb_sa_yield;
  localparam int N = 20000;
  real  v_bl, v_blb, t_res, v_os;
  logic sae;
  wire  dout;
  integer fout, k, nfail;
  integer seed = `SA_SEED;
  real dv;   // bitline signal: 5, 10, ... 50 mV, then 60 and 70 mV

  dram_sa_rnm #(.T0(`SA_T0), .TAU(`SA_TAU), .VSW(`SA_VSW), .TMIN(`SA_TMIN)) dut (
    .v_bl(v_bl), .v_blb(v_blb), .v_os(v_os), .sae(sae), .dout(dout), .t_res(t_res));

  // uniform in (0,1) from the seeded $random stream
  function automatic real uni();
    integer r;
    r = $random(seed);
    if (r < 0) r = -r;
    return ((r % 1000000) + 1) / 1000001.0;
  endfunction

  // Box-Muller normal sample
  function automatic real gauss(input real sigma);
    real a, b;
    a = uni();
    b = uni();
    return sigma * $sqrt(-2.0 * $ln(a)) * $cos(6.283185307179586 * b);
  endfunction

  initial begin
    sae = 0;
    fout = $fopen("rnm_yield.csv", "w");
    $fwrite(fout, "dv_V,n_reads,n_fail\n");
    for (int i = 0; i < 12; i++) begin
      dv = (i < 10) ? 0.005 * (i + 1) : 0.010 * (i - 4);
      nfail = 0;
      for (k = 0; k < N; k++) begin
        v_os  = gauss(`SA_SIGMA_OS);
        v_bl  = 0.9 + dv / 2.0;
        v_blb = 0.9 - dv / 2.0;
        #1 sae = 1;
        #20;                                  // longer than any resolve time
        if (dout !== 1'b1) nfail++;           // stored '1' must read as 1
        sae = 0;
      end
      $fwrite(fout, "%g,%0d,%0d\n", dv, N, nfail);
    end
    $fclose(fout);
    $finish;
  end
endmodule
