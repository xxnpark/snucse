`timescale 1ns / 1ps

module MUX_two(
    input [1:0] I0,
	input [1:0] I1,
	input S0,
	output [1:0] Z
	);
	
	assign Z = S0 ? I1 : I0;

endmodule
