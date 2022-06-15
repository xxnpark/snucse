`timescale 1ns / 1ps

module ALU(
    input signed [7:0] A,
	input signed [7:0] B,
	input ALUOp,
	output signed [7:0] O
	);
	
	assign O = ALUOp ? A + B : A + B;

endmodule
