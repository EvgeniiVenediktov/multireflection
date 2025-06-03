from matplotlib.pyplot import pyplot as plt 


sample = "2025-06-03 15:10:24,583 - INFO - origin_x:4.0,origin_y:2.0,adj_n:6,pred_x:0.03757834434509277,pred_y:0.038370609283447266,pos_x:-0.02,pos_y:-0.12,sim_index:0.92"

fname = "eval.log"

def process_line(s:str):
    s = s.split(" - INFO - ")[1]
    vals = s.split(",")
    assert(len(vals) == 8)
    origin_x  = float(vals[0].split(':')[1]) 
    origin_y  = float(vals[1].split(':')[1]) 
    adj_n     = float(vals[2].split(':')[1])    
    pred_x    = float(vals[3].split(':')[1])    
    pred_y    = float(vals[4].split(':')[1])   
    pos_x     = float(vals[5].split(':')[1])     
    pos_y     = float(vals[6].split(':')[1])  
    sim_index = float(vals[7].split(':')[1]) 



