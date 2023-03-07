import numpy as np
from gridworld import GridWorld
import random
from matplotlib import pyplot as plt

#Policy Iteration Algorithm
class PolicyIteration:

    def __init__(self, env, theta = 1e-6, gamma = 0.95):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'V': [],
            'iters': []
        }

    def policy_iteration(self,verbose = True, plots = True):
        V = np.zeros(self.env.num_states)
        pi = np.ones([self.env.num_states,self.env.num_actions])/self.env.num_actions
        iters = 0
        while True:
            delta = np.inf
            while delta > self.theta:
                # V = np.zeros(num_states)
                delta = 0
                iters += 1
                for s in range(self.env.num_states):
                    v = 0
                    for act,action_prob in enumerate(pi[s]):
                        for s_new in range(self.env.num_states):
                            v += self.env.p(s_new,s,act)*action_prob*(self.env.r(s,act) + self.gamma*V[s_new])
                    delta = max(delta, np.abs(v-V[s]))
                    V[s] = v
            self.log['V'].append(np.mean(V))
            self.log['iters'].append(iters)
            policy_stable = True
            for s in range(self.env.num_states):
                chosen_a = np.argmax(pi[s])
                pi_new = np.zeros(self.env.num_actions)
                for a in range(self.env.num_actions):
                    for s_new in range(self.env.num_states):
                        pi_new[a] += self.env.p(s_new,s,a)*(self.env.r(s,a) + self.gamma*V[s_new])
                best_a = np.argmax(pi_new)
                if chosen_a != best_a:
                    policy_stable = False
                pi[s] = np.eye(self.env.num_actions)[best_a]
            if policy_stable:
                if verbose == True:
                    print("Policy Iteration")
                    print("------------------------------------------------------")
                    print(f"Value function = {V}")
                    print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
                    print("------------------------------------------------------")
                if plots == True:
                    sp = self.env.reset()
                    self.log['s'].append(sp)
                    done = False
                    while not done:
                        a = np.argmax(pi[sp])
                        (sp, rp, done) = self.env.step(a)
                        self.log['t'].append(self.log['t'][-1] + 1)
                        self.log['s'].append(sp)
                        self.log['a'].append(a)
                        self.log['r'].append(rp)
                    
                    plt.figure()
                    plt.plot(self.log['t'], self.log['s'])
                    plt.plot(self.log['t'][:-1], self.log['a'])
                    plt.plot(self.log['t'][:-1], self.log['r'])
                    plt.title("State, Action and Reward for Policy Iteration")
                    plt.xlabel("Time")
                    plt.legend(['s', 'a', 'r'])
                    plt.savefig('figures/gridworld/trajectories_PI.png')

                    plt.figure()
                    plt.plot(self.log['iters'],self.log['V'])
                    plt.xlabel("Number of Iterations")
                    plt.ylabel("Mean of Value Function")
                    plt.title("Learning curve for number of iterations: Policy Iteration")
                    plt.savefig('figures/gridworld/learning_curve_PI.png')
                return V, pi, self.log


#Value Iteration Algorithm
class ValueIteration:

    def __init__(self, env, theta=1e-6, gamma=0.95):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'V': [],
            'iters': []
        }

    def value_iteration(self, verbose = True, plots = True):
        V = np.zeros(self.env.num_states)
        pi = np.ones([self.env.num_states,self.env.num_actions])/self.env.num_actions
        delta = np.inf
        iters = 0
        while delta > self.theta:
            delta = 0
            iters += 1
            for s in range(self.env.num_states):
                v = V[s]
                V_temp = np.zeros(self.env.num_actions)
                for a in range(self.env.num_actions):
                    for s_new in range(self.env.num_states):
                        V_temp[a] +=  self.env.p(s_new,s,a)*(self.env.r(s,a) + self.gamma*V[s_new])
                V[s] = np.max(V_temp)
                pi[s] = np.eye(self.env.num_actions)[np.argmax(V_temp)]
                delta = max(delta, np.abs(v-V[s]))
            self.log['V'].append(np.mean(V))
            self.log['iters'].append(iters)
        if verbose == True:
            print("Value Iteration")
            print("------------------------------------------------------")
            print(f"Value Function = {V}")
            print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
            print("------------------------------------------------------")
        
        if plots == True:
            sp = self.env.reset()
            self.log['s'].append(sp)
            done = False
            while not done:
                a = np.argmax(pi[sp])
                (sp, rp, done) = self.env.step(a)
                self.log['t'].append(self.log['t'][-1] + 1)
                self.log['s'].append(sp)
                self.log['a'].append(a)
                self.log['r'].append(rp)
            
            plt.figure()
            plt.plot(self.log['t'], self.log['s'])
            plt.plot(self.log['t'][:-1], self.log['a'])
            plt.plot(self.log['t'][:-1], self.log['r'])
            plt.title("State, Action and Reward for Value Iteration")
            plt.xlabel("Time")
            plt.legend(['s', 'a', 'r'])
            plt.savefig('figures/gridworld/trajectories_VI.png')

            plt.figure()
            plt.plot(self.log['iters'],self.log['V'])
            plt.xlabel("Number of Iterations")
            plt.ylabel("Mean of Value Function")
            plt.title("Learning curve for number of iterations: Value Iteration")
            plt.savefig('figures/gridworld/learning_curve_VI.png')

        return V,pi,self.log


#function for epsilon-greedy policy
def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action


#function for TD_0 algorithm
def TD_0(env, pi, alpha, num_episodes):
    num_states = 25
    V = np.zeros(num_states)
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': []
    }
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V


#SARSA Algorithm
class SARSA:
    def __init__(self,env,alpha=0.5,epsilon=0.1,num_episodes = 5000,gamma=0.95):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'G': [],
            'episodes': [],
            'iters': []
    }
    
    def SARSA(self,verbose = True,plots = True):
        Q = np.zeros((self.env.num_states,self.env.num_actions))
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions
        for episode in range(self.num_episodes):
            s = self.env.reset()
            a = epsilon_greedy(Q[s],self.epsilon)
            done = False
            G = 0
            iters = 0
            while not done:
                (s_new,r,done) = self.env.step(a)
                iters += 1
                G += r*self.gamma**(iters-1)
                a_new = epsilon_greedy(Q[s_new],self.epsilon)
                Q[s,a] += self.alpha*(r + self.gamma*Q[s_new,a_new] - Q[s,a])
                s = s_new
                a = a_new
            pi[s] = np.eye(self.env.num_actions)[np.argmax(Q[s])]
            self.log['G'].append(G)
            self.log['episodes'].append(episode)
        V_approx = TD_0(self.env, pi, self.alpha, self.num_episodes)
        if verbose == True:
            print(f"SARSA (Episodes = {self.num_episodes})")
            print("------------------------------------------------------")
            print(f"Q function = {Q}")
            print(f"Value function = {np.max(Q,axis=1)}")
            print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
            print(f"Approximate Value function using TD(0) = {V_approx}")
            print("------------------------------------------------------")
        if plots == True:
            sp = self.env.reset()
            self.log['s'].append(sp)
            done = False
            while not done:
                a = np.argmax(pi[sp])
                (sp, rp, done) = self.env.step(a)
                self.log['t'].append(self.log['t'][-1] + 1)
                self.log['s'].append(sp)
                self.log['a'].append(a)
                self.log['r'].append(rp)
            
            plt.figure()
            plt.plot(self.log['t'], self.log['s'])
            plt.plot(self.log['t'][:-1], self.log['a'])
            plt.plot(self.log['t'][:-1], self.log['r'])
            plt.title("State, Action and Reward for SARSA")
            plt.xlabel("Time")
            plt.legend(['s', 'a', 'r'])
            plt.savefig('figures/gridworld/trajectories_SARSA.png')

            plt.figure()
            plt.plot(self.log['episodes'],self.log['G'])
            plt.xlabel("Number of Episodes")
            plt.ylabel("Total Return (G)")
            plt.title("Learning curve for number of iterations: SARSA")
            plt.savefig('figures/gridworld/learning_curve_SARSA.png')

            plt.figure()
            plt.plot(V_approx)
            plt.xlabel("States")
            plt.ylabel("Value Function")
            plt.title("State-Value Function Learned by TD(0): SARSA")
            plt.savefig('figures/gridworld/state_value_SARSA.png')

            plt.figure()
        return Q, pi, self.log


#Q-Learning Algorithm
class QLearning:

    def __init__(self,env,alpha=0.5,epsilon=0.1,num_episodes = 5000,gamma=0.95):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'G': [],
            'episodes': [],
            'iters': []
    }
    def Q_Learning(self,verbose = True, plots = True):
        Q = np.zeros((self.env.num_states,self.env.num_actions))
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions
        for episode in range(self.num_episodes):
            s = self.env.reset()
            done = False
            G = 0
            iters = 0
            while not done:
                a = epsilon_greedy(Q[s],self.epsilon)
                (s_new,r,done) = self.env.step(a)
                iters += 1
                G += r*self.gamma**(iters-1)
                Q[s,a] += self.alpha*(r + self.gamma*np.max(Q[s_new]) - Q[s,a])
                s = s_new
            pi[s] = np.eye(self.env.num_actions)[np.argmax(Q[s])]
            self.log['G'].append(G)
            self.log['episodes'].append(episode)
        V_approx = TD_0(self.env, pi, self.alpha, self.num_episodes)
        if verbose == True:
            print(f"Q Learning (Episodes = {self.num_episodes})")
            print("------------------------------------------------------")
            print(f"Q function = {Q}")
            print(f"Value function = {np.max(Q,axis=1)}")
            print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
            print(f"Approximate Value function using TD(0) = {V_approx}")
            print("------------------------------------------------------")
        if plots == True:
            sp = self.env.reset()
            self.log['s'].append(sp)
            done = False
            while not done:
                a = np.argmax(pi[sp])
                (sp, rp, done) = self.env.step(a)
                self.log['t'].append(self.log['t'][-1] + 1)
                self.log['s'].append(sp)
                self.log['a'].append(a)
                self.log['r'].append(rp)
            
            plt.figure()
            plt.plot(self.log['t'], self.log['s'])
            plt.plot(self.log['t'][:-1], self.log['a'])
            plt.plot(self.log['t'][:-1], self.log['r'])
            plt.title("State, Action and Reward for Q Learning")
            plt.xlabel("Time")
            plt.legend(['s', 'a', 'r'])
            plt.savefig('figures/gridworld/trajectories_qlearning.png')

            plt.figure()
            plt.plot(self.log['episodes'],self.log['G'])
            plt.xlabel("Number of Episodes")
            plt.ylabel("Total Return (G)")
            plt.title("Learning curve for number of iterations: Q Learning")
            plt.savefig('figures/gridworld/learning_curve_qlearning.png')

            plt.figure()
            plt.plot(V_approx)
            plt.xlabel("States")
            plt.ylabel("Value Function")
            plt.title("State-Value Function Learned by TD(0): Q Learning")
            plt.savefig('figures/gridworld/state_value_qlearning.png')

            plt.figure()
        return Q, pi, self.log


def main():
    env = GridWorld(hard_version=False)
    env.reset()

    policy_iteration = PolicyIteration(env)
    V1, pi1, log1 = policy_iteration.policy_iteration(verbose=True, plots=True)

    value_iteration = ValueIteration(env)
    V2, pi2, log2 = value_iteration.value_iteration(verbose=True, plots=True)

    sarsa = SARSA(env)
    V3, pi3, log3 = sarsa.SARSA(verbose=True, plots=True)

    qlearning = QLearning(env)
    V4, pi4, log4 = qlearning.Q_Learning(verbose=True, plots=True)

    algorithms = {0:"Policy Iteration", 1:"Value Iteration", 2:"SARSA",3:"Q-Learning"}
    #plotting alpha and epsilon sweep plots
    for i in range(2):
        if i == 0:
            alpha_vals = np.linspace(0,1,11)
            plt.figure()
            for alpha in alpha_vals:
                sarsa_alpha = SARSA(env,alpha)
                Q_alpha, pi_alpha, log_alpha = sarsa_alpha.SARSA(verbose=False, plots = False)
                plt.plot(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
            plt.title(f"Learning curve for different alpha: {algorithms[i+2]}")
            plt.legend()
            plt.savefig('figures/gridworld/alpha_sweep_SARSA.png')

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                sarsa_epsilon = SARSA(env, epsilon = epsilon)
                Q_eps, pi_eps, log_eps = sarsa_epsilon.SARSA(verbose=False, plots = False)
                plt.plot(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
            plt.title(f"Learning curve for different epsilon: {algorithms[i+2]}")
            plt.legend()
            plt.savefig('figures/gridworld/epsilon_sweep_SARSA.png')
        
        else:
            alpha_vals = np.linspace(0,1,11)
            plt.figure()
            for alpha in alpha_vals:
                qlearning_alpha = QLearning(env, alpha=alpha)
                Q_alpha, pi_alpha, log_alpha = qlearning_alpha.Q_Learning(verbose=False, plots = False)
                plt.plot(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
            plt.title(f"Learning curve for different alpha: {algorithms[i+2]}")
            plt.legend()
            plt.savefig('figures/gridworld/alpha_sweep_qlearning.png')

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                qlearning_epsilon = QLearning(env, epsilon=epsilon)
                Q_eps, pi_eps, log_eps = qlearning_epsilon.Q_Learning(verbose=False, plots=False)
                plt.plot(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
            plt.title(f"Learning curve for different epsilon: {algorithms[i+2]}")
            plt.legend()
            plt.savefig('figures/gridworld/epsilon_sweep_qlearning.png')
            


if __name__ == '__main__':
    main()