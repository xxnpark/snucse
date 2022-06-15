`timescale 1ns / 1ps

module MUX_eight(
    input [7:0] I0,
	input [7:0] I1,
	input S0,
	output [7:0] Z
	);
	
	assign Z = S0 ? I1 : I0;

endmodule