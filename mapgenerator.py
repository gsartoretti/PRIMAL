from tkinter import *
import random
import math
import time
from matplotlib.colors import hsv_to_rgb
import numpy as np
import os

DYNAMIC_TESTING=True
GOALS=True
output_path="environments"
model_path="model_primal"
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}

if DYNAMIC_TESTING:
    import tensorflow as tf
    from ACNet import ACNet
    
def init(data):
    data.size=10
    data.state=np.zeros((data.size,data.size)).astype(int)
    data.goals=np.zeros((data.size,data.size)).astype(int)
    data.mode="obstacle"
    data.agent_counter=1
    data.primed_goal=0
    data.ID=0
    data.paused=True
    data.blocking_confidences=[]
    data.agent_goals=[]
    data.agent_positions=[]    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(output_path):
        for (_,_,files) in os.walk(output_path):
            for f in files:
                if ".npy" in f:
                    try:
                        ID=int(f[:f.find(".npy")])
                    except Exception:
                        continue
                    if ID>data.ID:
                        data.ID=ID
    data.ID+=1
    if DYNAMIC_TESTING:
        data.rnn_states=[]
        data.sess=tf.Session()
        data.network=ACNet("global",5,None,False,10,"global")
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(data.sess,ckpt.model_checkpoint_path)        
        
def getDir(action):
    return dirDict[action]

def mousePressed(event, data):
    r=int((event.y/data.height)*data.state.shape[0])
    c=int((event.x/data.width)*data.state.shape[1])    
    if data.mode=="obstacle":
        if data.state[r,c]<=0 and data.goals[r,c]==0:
            data.state[r,c]=-((data.state[r,c]+1)%2)
    elif data.mode=="agent":
        if data.state[r,c]==0:
            data.state[r,c]=data.agent_counter
            data.goals[r,c]=data.agent_counter
            data.agent_positions.append((r,c))
            data.blocking_confidences.append(0)            
            data.rnn_states.append(data.network.state_init)
            data.agent_goals.append((r,c))
            data.agent_counter+=1
    elif data.mode=="goal":
        if data.state[r,c]>0 and data.primed_goal==0:
            data.primed_goal=data.state[r,c]
        elif data.state[r,c]!=-1 and data.primed_goal>0 and data.goals[r,c]==0:
            removeGoal(data,data.primed_goal)
            data.agent_goals[data.primed_goal-1]=(r,c)
            data.goals[r,c]=data.primed_goal
            data.primed_goal=0
            
def removeGoal(data,agent):
    for i in range(data.state.shape[0]):
        for j in range(data.state.shape[1]):
            if data.goals[i,j]==agent:
                data.goals[i,j]=0
                
def keyPressed(event, data):
    if event.keysym=='r':
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)
        data.agent_goals=[]
        data.rnn_states=[]
        data.agent_positions=[]        
        data.blocking_confidences=[]        
        data.primed_goal=0
        data.agent_counter=1
    elif event.keysym=="c":
        data.agent_counter=1
        data.primed_goal=0
        data.rnn_states=[]
        data.blocking_confidences=[]        
        data.agent_goals=[]
        data.agent_positions=[]
        data.goals=np.zeros((data.size,data.size))
        data.state=-(data.state==-1).astype(int)
    elif event.keysym=="p":
        data.paused=not data.paused
    elif event.keysym=="o":
        data.mode="obstacle"
    elif event.keysym=="g":
        data.mode="goal"
    elif event.keysym=="a":
        data.mode="agent"
    elif event.keysym=='Up':
        data.size+=1
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)
    elif event.keysym=='Down':
        data.size-=1;
        if data.size<1:
            data.size==1
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)        
    elif event.keysym=="s":
        savedata=np.array([data.state,data.goals,data.agent_counter-1])
        np.save(output_path+"/%d"%data.ID,savedata)
        data.ID+=1
        
def observe(data,agent_id,goals):
    assert(agent_id>0)
    top_left=(data.agent_positions[agent_id-1][0]-10//2,data.agent_positions[agent_id-1][1]-10//2)
    bottom_right=(top_left[0]+10,top_left[1]+10)        
    obs_shape=(10,10)
    goal_map             = np.zeros(obs_shape)
    poss_map             = np.zeros(obs_shape)
    obs_map              = np.zeros(obs_shape)
    goals_map            = np.zeros(obs_shape)
    visible_agents=[]    
    for i in range(top_left[0],top_left[0]+10):
        for j in range(top_left[1],top_left[1]+10):
            if i>=data.state.shape[0] or i<0 or j>=data.state.shape[1] or j<0:
                #out of bounds, just treat as an obstacle
                obs_map[i-top_left[0],j-top_left[1]]=1
                continue
            if data.state[i,j]==-1:
                #obstacles
                obs_map[i-top_left[0],j-top_left[1]]=1
            if data.state[i,j]==agent_id:
                #agent's position
#                     pos_map[i-top_left[0],j-top_left[1]]=1
                poss_map[i-top_left[0],j-top_left[1]]=1
            elif data.goals[i,j]==agent_id:
                #agent's goal
                goal_map[i-top_left[0],j-top_left[1]]=1
            if data.state[i,j]>0 and data.state[i,j]!=agent_id:
                #other agents' positions
                poss_map[i-top_left[0],j-top_left[1]]=1
                visible_agents.append(data.state[i,j])                
    dx=data.agent_goals[agent_id-1][0]-data.agent_positions[agent_id-1][0]
    dy=data.agent_goals[agent_id-1][1]-data.agent_positions[agent_id-1][1]
    mag=(dx**2+dy**2)**.5
    if mag!=0:
        dx=dx/mag
        dy=dy/mag
    if goals:
        distance=lambda x1,y1,x2,y2:((x2-x1)**2+(y2-y1)**2)**.5
        for agent in visible_agents:
            x,y=data.agent_goals[agent-1]
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                #out of observation
                min_node=(-1,-1)
                min_dist=1000
                for i in range(top_left[0],top_left[0]+10):
                    for j in range(top_left[1],top_left[1]+10):
                        d=distance(i,j,x,y)
                        if d<min_dist:
                            min_node=(i,j)
                            min_dist=d
                goals_map[min_node[0]-top_left[0],min_node[1]-top_left[1]]=1
            else:
                goals_map[x-top_left[0],y-top_left[1]]=1
        return  ([poss_map,goal_map,goals_map,obs_map],[dx,dy,mag])
    else:
        return ([poss_map,goal_map,obs_map],[dx,dy,mag])

def timerFired(data):
    if DYNAMIC_TESTING and not data.paused:
        for (x,y) in data.agent_positions:
            ID=data.state[x,y]
            observation=observe(data,ID,GOALS)
            rnn_state=data.rnn_states[ID-1]#yes minus 1 is correct
            a_dist,v,rnn_state,blocking = data.sess.run([data.network.policy,data.network.value,data.network.state_out,data.network.blocking], 
                                                   feed_dict={data.network.inputs:[observation[0]],
                                                            data.network.goal_pos:[observation[1]],
                                                            data.network.state_in[0]:rnn_state[0],
                                                            data.network.state_in[1]:rnn_state[1]})
            data.rnn_states[ID-1]=rnn_state      
            data.blocking_confidences[ID-1]=np.ravel(blocking)[0]
            action=np.argmax(a_dist)
            dx,dy =getDir(action)
            ax,ay =data.agent_positions[ID-1]
            if(ax+dx>=data.state.shape[0] or ax+dx<0 or ay+dy>=data.state.shape[1] or ay+dy<0):#out of bounds
                continue
            if(data.state[ax+dx,ay+dy]<0):#collide with static obstacle
                continue
            if(data.state[ax+dx,ay+dy]>0):#collide with robot
                continue
            # No collision: we can carry out the action
            data.state[ax,ay] = 0
            data.state[ax+dx,ay+dy] = ID
            data.agent_positions[ID-1] = (ax+dx,ay+dy)            

def redrawAll(canvas, data):
    for r in range(data.state.shape[0]):
        y=(data.height/data.state.shape[0])*r
        color_depth=30
        for c in range(data.state.shape[1]):
            x=(data.height/data.state.shape[0])*c
            if data.state[r,c]==-1:
                canvas.create_rectangle(x, y, x+data.width/data.state.shape[0], y+data.height/data.state.shape[1],
                                            fill='grey', width=0)
            elif data.state[r,c]>0:
                color=hsv_to_rgb(np.array([(data.state[r,c]%color_depth)/float(color_depth),1,1]))
                color*=255
                color=color.astype(int)
                mycolor = '#%02x%02x%02x' % (color[0], color[1], color[2])
                canvas.create_rectangle(x, y, x+data.width/data.state.shape[0], y+data.height/data.state.shape[1],
                                                    fill=mycolor, width=0)  
                confidence=data.blocking_confidences[data.state[r,c]-1]
                confidence="%.0001f"%confidence
                canvas.create_text(x+data.width/data.state.shape[0]/2, y+data.height/data.state.shape[1]/2,
                                                   fill='black', anchor="s",text=confidence,font="Arial 30 bold")                 
            if data.goals[r,c]>0:
                color=hsv_to_rgb(np.array([(data.goals[r,c]%color_depth)/float(color_depth),1,1]))
                color*=255
                color=color.astype(int)
                mycolor = '#%02x%02x%02x' % (color[0], color[1], color[2])
                if data.state[r,c]==data.goals[r,c]:
                    canvas.create_text(x+data.width/data.state.shape[0]/2, y+data.height/data.state.shape[1]/2,
                                                        fill="black", anchor="center",text="+",font="Arial 50 bold")
                else:
                    canvas.create_text(x+data.width/data.state.shape[0]/2, y+data.height/data.state.shape[1]/2,
                                                       fill=mycolor, anchor="center",text="+",font="Arial 50 bold")      
    for r in range(data.state.shape[0]):
        y=(data.height/data.state.shape[0])*r
        canvas.create_line(0,y,data.width,y,fill="black")
    for c in range(data.state.shape[1]):
        x=(data.height/data.state.shape[0])*c
        canvas.create_line(x,0,x,data.height,fill="black")
    canvas.create_text(data.width/2, 20,
                                fill="black", text=data.mode,font="Arial 20",anchor="center")
    txt="Paused" if data.paused else "Running"
    canvas.create_text(data.width-100, 20,
                       fill="black", text=txt,font="Arial 20",anchor="center")    
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 200 # milliseconds
    init(data)
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

def main():
    run(900,900)
if __name__=='__main__':
    main()
