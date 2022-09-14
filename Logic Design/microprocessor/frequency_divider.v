`timescale 1ns / 1ps

module frequency_divider(
    input clkin,
    input clr,
    output reg clkout
    );
	 
	reg [31:0] cnt;
	initial begin
	cnt = 0;
	clkout = 0;
	end
	
	always @(posedge clkin or posedge clr)
		begin
			if(clr)
				begin
					cnt <= 32'd0;
					clkout <= 1'b0;
				end
			else if(cnt == 32'd25000000) // d25000000
				begin
					cnt <= 32'd0;
					clkout <= ~clkout;
				end
			else
				begin
					cnt <= cnt + 1;
				end
		end

endmodule
