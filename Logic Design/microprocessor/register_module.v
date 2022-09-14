`timescale 1ns / 1ps

module register_module(
input [1:0] read_register_one,
	input [1:0] read_register_two,
	input [1:0] write_register,
	input [7:0] write_data,
	input RegWrite,
	input CLK,
	input reset,
	output [7:0] read_data_one,
	output [7:0] read_data_two
	);
	
	reg [7:0] registers [0:3];
	
	integer i;
	initial begin
	for (i = 0; i < 4; i = i + 1)
		registers[i] <= 0;
	end
	
	assign read_data_one = registers[read_register_one];
	assign read_data_two = registers[read_register_two];
	
	always @(posedge CLK or posedge reset)
		begin
			if (reset)
				begin
					for (i = 0; i < 4; i = i + 1)
						registers[i] <= 0;
				end
			else
				begin
					if (RegWrite)
						registers[write_register] <= write_data;
				end
		end

endmodule
