`timescale 1ns / 1ps

module control_unit(
	input [1:0] op,
	output RegDst,
	output RegWrite,
	output ALUSrc,
	output Branch,
	output MemRead,
	output MemWrite,
	output MemtoReg,
	output ALUOp
    );
	 
	
	reg [7:0] out;
	assign {RegDst, RegWrite, ALUSrc, Branch, MemRead, MemWrite, MemtoReg, ALUOp} = out; // assume DC terms as 0
	
	always @(op)
		begin
			case (op)
				0: out = 8'b11000001;
				1: out = 8'b01101010;
				2: out = 8'b00100100;
				3: out = 8'b00010000;
			endcase
		end

endmodule
