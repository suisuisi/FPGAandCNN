module memorywork(clk,data,address,we_p,we_w,re_RAM,nextstep,dp,dw,addrp,addrw,step_out,GO,in_dense);

parameter num_conv=0;

parameter picture_size=0;
parameter convolution_size=0;
parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_5=0;
parameter SIZE_6=0;
parameter SIZE_7=0;
parameter SIZE_8=0;
parameter SIZE_9=0;
parameter SIZE_address_pix=0;
parameter SIZE_address_wei=0;

inout clk;
input signed [SIZE_1-1:0] data;
output [12:0] address;
output reg we_p;
output reg we_w;
inout re_RAM;
input nextstep;
output reg signed [SIZE_1-1:0] dp;   //write data
output reg signed [SIZE_9-1:0] dw;     //write weight
output reg [SIZE_address_pix-1:0] addrp;
output reg [SIZE_address_wei-1:0] addrw;
output [4:0] step_out;
input GO;
input [4:0] in_dense;
		  
reg [SIZE_address_pix-1:0] addr;
wire [12:0] firstaddr,lastaddr;
reg sh;

reg [4:0] step;
reg [4:0] step_n;
reg [4:0] weight_case;
reg [SIZE_9-1:0] buff;
reg [12:0] i;
reg [12:0] i_d;
reg [12:0] i1;
addressRAM #(.picture_size(picture_size), .convolution_size(convolution_size)) inst_1(.step(step_out),.re_RAM(re_RAM),.firstaddr(firstaddr),.lastaddr(lastaddr));
initial sh=0;
initial weight_case=0;
initial i=0;
initial i_d=0;
initial i1=0;
always @(posedge clk)
    begin
		  if (GO==1) step=1;
		  sh=sh+1;
				if (step_out==1)
				   begin
					if ((i<=lastaddr-firstaddr)&&(sh==0))
                    begin
                        //address=firstaddr+i;
								addr=i;
								if (step_out==1) we_p=1;
					end
                if ((i<=lastaddr-firstaddr)&&(sh==1))
                    begin
                        if (we_p) 
									begin
											addrp=addr;
											dp=0;
											dp[SIZE_1-1:0]=data;
											we_p=0;
									end
									i=i+1;
							end
					 if ((i>lastaddr-firstaddr)&&(sh==1))
                    begin
                        step=step+1;          //next step
                        i=0;
                    end
				end
				if ((step_out==2)||(step_out==4)||(step_out==6)||(step_out==8)||(step_out==10)||(step_out==12)||(step_out==14))
					begin
						if ((i<=lastaddr-firstaddr)&&(sh==0))
                    begin
								addr=i1;
						  end
						if ((i<=lastaddr-firstaddr)&&(sh==1))
                    begin
										we_w=0;
										addrw=addr;
										if (weight_case!=0) i=i+1; 
										if (step_out==14) if (i_d==(in_dense)) begin  dw=buff; we_w=1; weight_case=1; i_d=0; i1=i1+1; end
										case (weight_case)
											0: ;
											1: begin buff=0; buff[SIZE_9-1:SIZE_8]=data; end   
											2: buff[SIZE_8-1:SIZE_7]=data; 
											3: buff[SIZE_7-1:SIZE_6]=data;  
											4: buff[SIZE_6-1:SIZE_5]=data;  
											5: buff[SIZE_5-1:SIZE_4]=data;           
											6: buff[SIZE_4-1:SIZE_3]=data;  
											7: buff[SIZE_3-1:SIZE_2]=data;  
											8: buff[SIZE_2-1:SIZE_1]=data;   
											9: begin buff[SIZE_1-1:0]=data;  i1=i1+1; end
											default: $display("Check weight_case");
										endcase
										if (weight_case!=0) i_d=i_d+1;
										if (weight_case==9) begin weight_case=1; dw=buff; we_w=1; end else weight_case=weight_case+1;
										
                    end
						if ((i>lastaddr-firstaddr)&&(sh==1))
                    begin
                        step=step+1;          //next step
                        i=0;
								i_d=0;
								i1=0;
								weight_case=0;
                    end
            end
				else
					we_w=0;
    end
always @(posedge nextstep) if (GO==1) step_n=0; else step_n=step_n+1;
assign step_out=step+step_n;
assign address=firstaddr+i;
endmodule
