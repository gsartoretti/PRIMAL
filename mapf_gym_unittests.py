import unittest
import mapf_gym as MAPF_Env
import numpy as np


# Agent 1
num_agents1 = 1
world1 = [[ 1,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals1 = [[ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

# Agent 1
num_agents2 = 1
world2 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0, -1,  1, -1,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals2 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

# Agent 1 and 2
num_agents3 = 2
world3 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1, -1,  0,  0,  0,  0,  0],
          [ 0,  0, -1,  1,  2, -1,  0,  0,  0,  0],
          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals3 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

# Agent 1 and 2
num_agents4 = 2
world4 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1, -1,  -1,  0,  0,  0,  0],
          [ 0,  0, -1,  1,  2, -1,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals4 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]


# action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_south, 4:MOVE_WEST}
# MAPF_Env.ACTION_COST, MAPF_Env.IDLE_COST, MAPF_Env.GOAL_REWARD, MAPF_Env.COLLISION_REWARD
FULL_HELP = False

class MAPFTests(unittest.TestCase):
    # Bruteforce tests
    def test_validActions1(self):
        # MAPF_Env.MAPFEnv(self, num_agents=1, world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=10, PROB=.2, FULL_HELP=False)
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1), DIAGONAL_MOVEMENT=False)
        validActions1 = gameEnv1._listNextValidActions(1)
        self.assertEqual(validActions1, [0,1,2])
        # With diagonal actions
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1), DIAGONAL_MOVEMENT=True)
        validActions1 = gameEnv1._listNextValidActions(1)
        self.assertEqual(validActions1, [0,1,2,5])
        
    def test_validActions2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2), DIAGONAL_MOVEMENT=False)
        validActions2 = gameEnv2._listNextValidActions(1)
        self.assertEqual(validActions2, [0])
        # With diagonal actions
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2), DIAGONAL_MOVEMENT=True)
        validActions2 = gameEnv2._listNextValidActions(1)
        self.assertEqual(validActions2, [0,5,6,7,8])
    
    def test_validActions3(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3), DIAGONAL_MOVEMENT=False)
        validActions3a = gameEnv3._listNextValidActions(1)
        validActions3b = gameEnv3._listNextValidActions(2)
        self.assertEqual(validActions3a, [0])
        self.assertEqual(validActions3b, [0,2])
        # With diagonal actions
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3), DIAGONAL_MOVEMENT=True)
        validActions3a = gameEnv3._listNextValidActions(1)
        validActions3b = gameEnv3._listNextValidActions(2)
        self.assertEqual(validActions3a, [0,5,6,7])
        self.assertEqual(validActions3b, [0,2,5,8])

    def test_validActions4(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=False)
        validActions4a = gameEnv4._listNextValidActions(1)
        validActions4b = gameEnv4._listNextValidActions(2)
        self.assertEqual(validActions4a, [0,2])
        self.assertEqual(validActions4b, [0,2])    
        # With diagonal actions
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        validActions4a = gameEnv4._listNextValidActions(1)
        validActions4b = gameEnv4._listNextValidActions(2)
        self.assertEqual(validActions4a, [0,2,5,6,7])
        self.assertEqual(validActions4b, [0,2,5,6])    

        
    def testIdle1(self):
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal, blocking, valid_action
        s1, r, d, _, o_g, _, _ = gameEnv1.step((1,0))
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, MAPF_Env.IDLE_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def testIdle2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _, o_g, _, _ = gameEnv2.step((1,0))
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def testIdle3(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,0))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,0))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.IDLE_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def testIdle4(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=False)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,0))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,0))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.IDLE_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_east1(self):
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _, o_g, _, _ = gameEnv1.step((1,1))
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _, o_g, _, _ = gameEnv2.step((1,1))
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east3a(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,1))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,1))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east3b(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,1))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,1))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,1))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,1))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,1))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,1))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_north1(self):
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _, o_g, _, _ = gameEnv1.step((1,2))
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _, o_g, _, _ = gameEnv2.step((1,2))
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north3a(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,2))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,2))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north3b(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,2))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,2))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,2))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,2))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,2))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,2))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_west1(self):
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _, o_g, _, _ = gameEnv1.step((1,3))
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _, o_g, _, _ = gameEnv2.step((1,3))
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west3a(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,3))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,3))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west3b(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,3))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,3))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,3))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,3))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,3))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,3))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_south1(self):
        gameEnv1 = MAPF_Env.MAPFEnv(num_agents1, world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _, o_g, _, _ = gameEnv1.step((1,4))
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south2(self):
        gameEnv2 = MAPF_Env.MAPFEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _, o_g, _, _ = gameEnv2.step((1,4))
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south3a(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,4))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,4))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south3b(self):
        gameEnv3 = MAPF_Env.MAPFEnv(num_agents3, world0=np.array(world3), goals0=np.array(goals3))
        s0 = gameEnv3.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv3.step((2,4))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv3.step((1,4))
        s2 = gameEnv3.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,4))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,4))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4))
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,4))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,4))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_northeast4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_northeast4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_northwest4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_northwest4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_southwest4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_southwest4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_southeast4a(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,8))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,8))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_southeast4b(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,8))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,8))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))


    # Other Justin tests
    def test_move_diag1(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(2,gameEnv4.world.state[4,5])
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(1,gameEnv4.world.state[4,4])
        
    def test_move_diag2(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(2,gameEnv4.world.state[4,3])
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(1,gameEnv4.world.state[4,2])    
    def test_move_diag3(self):
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,8))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(2,gameEnv4.world.state[3,4])
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(1,gameEnv4.world.state[2,2])    
    def test_move_diag4(self):
        #test diag collisions
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(2,gameEnv4.world.state[4,3])
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(1,gameEnv4.world.state[3,3])    
    def test_move_diag5(self):
        #tests diag collisions
        gameEnv4 = MAPF_Env.MAPFEnv(num_agents4, world0=np.array(world4), goals0=np.array(goals4),DIAGONAL_MOVEMENT=True)
        s0 = gameEnv4.world.state.copy()
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,5))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(1,gameEnv4.world.state[4,4])
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(2,gameEnv4.world.state[3,4])    
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(2,gameEnv4.world.state[3,4])  
        # Agent 1
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,7))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.GOAL_REWARD)
        self.assertFalse(d)
        self.assertTrue(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))
        self.assertEqual(1,gameEnv4.world.state[3,3])
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(2,gameEnv4.world.state[3,4])  
        #after waiting, we should be able to cross diagonally
        s1, r, d, _, o_g, _, _ = gameEnv4.step((1,0))
        # Agent 2
        s1, r, d, _, o_g, _, _ = gameEnv4.step((2,6))
        s2 = gameEnv4.world.state.copy()
        self.assertEqual(r, MAPF_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertFalse(o_g)
        self.assertEqual(np.sum(s0), np.sum(s2))    
        self.assertEqual(2,gameEnv4.world.state[4,3])     
if __name__ == '__main__':
    unittest.main()
