`timescale 1ns / 1ps

module PC(
    input CLK,
	input reset,
	input [7:0] D,
	output reg [7:0] Q
	);
	
	initial Q = 0;
	
	always @(posedge CLK or posedge reset)
		begin
			if (reset)
				Q <= 0;
			else
				Q <= D;
		end

endmodule