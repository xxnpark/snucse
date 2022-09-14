`timescale 1ns / 1ps

module data_memory(
	input [7:0] address,
	input [7:0] write_data,
	input MemWrite, MemRead,
	input CLK,
	input reset,
	output reg [7:0] read_data
	);

	reg [7:0] MEMORY [0:31];
	integer i;
	
	initial begin
		for (i=0; i<32; i=i+1)
			MEMORY[i] = i < 16 ? i : 16 - i;
	end

	always @(posedge CLK or posedge reset)
		begin
			if	(reset)
				begin
					for (i=0; i<32; i=i+1)
						MEMORY[i] <= i < 16 ? i : 16 - i;
				end
			else
				begin
					if (MemWrite == 1)
						begin
							MEMORY[address] <= write_data;
						end
				end
		end
		
	always @(address or MemRead)
		begin
			if (MemRead == 1)
				read_data <= MEMORY[address];
		end
		

endmodule
