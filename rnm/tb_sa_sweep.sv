// Drives the RNM with the same bitline splits used in the SPICE sweep and
// records when the output resolves. Compared against SPICE by scripts/m5_rnm.py.
//   in : rnm/dv_list.txt   (one dv in volts per line)
//   out: rnm/rnm_sweep.csv (dv_V, decision, t_res_ns)
`timescale 1ns/1fs
`include "sa_params.svh"

module tb_sa_sweep;
  real  v_bl, v_blb, t_res, dv, t_start;
  real  v_os = 0.0;
  logic sae;
  wire  dout;
  integer fin, fout, n;

  dram_sa_rnm #(.T0(`SA_T0), .TAU(`SA_TAU), .VSW(`SA_VSW), .TMIN(`SA_TMIN)) dut (
    .v_bl(v_bl), .v_blb(v_blb), .v_os(v_os), .sae(sae), .dout(dout), .t_res(t_res));

  initial begin
    sae = 0;
    fin  = $fopen("dv_list.txt", "r");
    fout = $fopen("rnm_sweep.csv", "w");
    $fwrite(fout, "dv_V,decision,t_res_ns\n");
    while ($fscanf(fin, "%f\n", dv) == 1) begin
      v_bl  = 0.9 + dv / 2.0;
      v_blb = 0.9 - dv / 2.0;
      #1 sae = 1;
      t_start = $realtime;
      // wait for a decision or give up after 20 ns
      fork : wait_res
        begin wait (dout === 1'b1 || dout === 1'b0); disable wait_res; end
        begin #20; disable wait_res; end
      join
      $fwrite(fout, "%g,%b,%g\n", dv, dout, (dout === 1'bx) ? -1.0 : $realtime - t_start);
      #1 sae = 0;
    end
    $fclose(fout);
    $finish;
  end
endmodule
