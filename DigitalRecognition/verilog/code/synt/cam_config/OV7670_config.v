`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/06/2015 02:09:28 PM
// Design Name: 
// Module Name: OV7670_config
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module OV7670_config
#(
    parameter CLK_FREQ = 25000000
)
(
    input wire clk,
    input wire SCCB_interface_ready,
    input wire [15:0] rom_data,
    input wire start,
    output reg [7:0] rom_addr,
    output reg done,
    output reg [7:0] SCCB_interface_addr,
    output reg [7:0] SCCB_interface_data,
    output reg SCCB_interface_start
    );
    
    initial begin
        rom_addr = 0;
        done = 0;
        SCCB_interface_addr = 0;
        SCCB_interface_data = 0;
        SCCB_interface_start = 0;
    end
    
    localparam FSM_IDLE = 0;
    localparam FSM_SEND_CMD = 1;
    localparam FSM_DONE = 2;
    localparam FSM_TIMER = 3;
    
    reg [2:0] FSM_state = FSM_IDLE;
    reg [2:0] FSM_return_state;
    reg [31:0] timer = 0; 
    
    always@(posedge clk) begin
    
        case(FSM_state)
            
            FSM_IDLE: begin 
                FSM_state <= start ? FSM_SEND_CMD : FSM_IDLE;
                rom_addr <= 1'h0;
                done <= start ? 1'h0 : done;
            end
            
            FSM_SEND_CMD: begin 
                case(rom_data)
                    16'hFFFF: begin //end of ROM
                        FSM_state <= FSM_DONE;
                    end
                    
                    16'hFFF0: begin //delay state 
                        timer <= (CLK_FREQ/100); //10 ms delay
                        FSM_state <= FSM_TIMER;
                        FSM_return_state <= FSM_SEND_CMD;
                        rom_addr <= rom_addr + 1'h1;
                    end
                    
                    default: begin //normal rom commands
                        if (SCCB_interface_ready) begin
                            FSM_state <= FSM_TIMER;
                            FSM_return_state <= FSM_SEND_CMD;
                            timer <= 1'h0; //one cycle delay gives ready chance to deassert
                            rom_addr <= rom_addr + 1'h1;
                            SCCB_interface_addr <= rom_data[15:8];
                            SCCB_interface_data <= rom_data[7:0];
                            SCCB_interface_start <= 1;
                        end
                    end
                endcase
            end
                        
            FSM_DONE: begin //signal done 
                FSM_state <= FSM_IDLE;
                done <= 1'h1;
            end
                           
                
            FSM_TIMER: begin //count down and jump to next state
                FSM_state <= (timer == 0) ? FSM_return_state : FSM_TIMER;
                timer <= (timer==0) ? 0 : ( timer - 1'h1 );
                SCCB_interface_start <= 1'h0;
            end
        endcase
    end
endmodule
