import numpy as np
from discrete_pendulum import Pendulum
import random
from matplotlib import pyplot as plt

def wrap_pi(x):
    theta = ((x + np.pi) % (2 * np.pi)) - np.pi
    return theta

def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action

def TD_0(env, pi, alpha, num_episodes):
    V = np.zeros(env.num_states)
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': [],
        'theta': [],
        'thetadot': []
    }
    for episode in range(num_episodes):
        s = env.reset()
        log['s'].append(s)
        log['theta'].append(wrap_pi(env.x[0]))
        log['thetadot'].append(env.x[1])
        done = False
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(wrap_pi(env.x[0]))
            log['thetadot'].append(env.x[1])
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V, log

class SARSA:
    def __init__(self,env,alpha=0.3,epsilon=0.8,num_episodes = 3000,gamma=0.95):
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
            'iters': [],
            'theta': [],
            'thetadot': []
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
        V_approx, log_TD_0 = TD_0(self.env, pi, self.alpha, self.num_episodes)

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
            self.log['theta'].append(wrap_pi(self.env.x[0]))
            self.log['thetadot'].append(self.env.x[1])
            done = False
            while not done:
                a = np.argmax(pi[sp])
                (sp, rp, done) = self.env.step(a)
                self.log['t'].append(self.log['t'][-1] + 1)
                self.log['s'].append(sp)
                self.log['a'].append(a)
                self.log['r'].append(rp)
                self.log['theta'].append(wrap_pi(self.env.x[0]))
                self.log['thetadot'].append(self.env.x[1])
            
            plt.figure()
            plt.plot(self.log['t'], self.log['s'])
            plt.plot(self.log['t'][:-1], self.log['a'])
            plt.plot(self.log['t'][:-1], self.log['r'])
            plt.title("State, Action and Reward for SARSA")
            plt.xlabel("Time")
            plt.legend(['s', 'a', 'r'])
            plt.savefig('figures/pendulum/trajectories_SARSA.png')

            plt.figure()
            plt.plot(self.log['t'], self.log['theta'])
            plt.plot(self.log['t'], self.log['thetadot'])
            plt.axhline(y=np.pi, color='r', linestyle='-')
            plt.axhline(y=-np.pi, color='r', linestyle='-')
            plt.title("Theta, Theta_dot vs Time for SARSA")
            plt.xlabel("Time")
            plt.legend(['theta','theta_dot'])
            plt.savefig('figures/pendulum/theta_thetadot_SARSA.png')

            plt.figure()
            plt.plot(self.log['episodes'],self.log['G'])
            plt.xlabel("Number of Episodes")
            plt.ylabel("Total Return (G)")
            plt.title("Learning curve for number of iterations: SARSA")
            plt.savefig('figures/pendulum/learning_curve_SARSA.png')

            plt.figure()
            plt.plot(V_approx)
            plt.xlabel("States")
            plt.ylabel("Value Function")
            plt.title("State-Value Function Learned by TD(0): SARSA")
            plt.savefig('figures/pendulum/state_value_SARSA.png')

            plt.figure()
            plt.plot(log_TD_0['t'],log_TD_0['theta'])
            plt.plot(log_TD_0['t'],log_TD_0['thetadot'])
            plt.title("Theta, Theta_dot vs Time for TD(0) : SARSA")
            plt.xlabel("Time")
            plt.legend(['theta','theta_dot'])
            plt.savefig('figures/pendulum/theta_thetadot_SARSA_TD_0.png')
        return Q, pi, self.log


class QLearning:

    def __init__(self, env, alpha = 0.3, epsilon = 0.8, num_episodes = 3000, gamma = 0.95):
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
            'iters': [],
            'theta': [],
            'thetadot': []
    }
        
    def Q_Learning(self, verbose=True, plots=True):
        gamma = 0.95
        Q = np.zeros((self.env.num_states,self.env.num_actions))
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions
        log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'G': [],
            'episodes': [],
            'theta': [],
            'thetadot': []
        }
        for episode in range(self.num_episodes):
            s = self.env.reset()
            done = False
            G = 0
            iters = 0
            while not done:
                a = epsilon_greedy(Q[s],self.epsilon)
                (s_new,r,done) = self.env.step(a)
                iters += 1
                G += r*gamma**(iters-1)
                Q[s,a] += self.alpha*(r + gamma*np.max(Q[s_new]) - Q[s,a])
                s = s_new
            pi[s] = np.eye(self.env.num_actions)[np.argmax(Q[s])]
            log['G'].append(G)
            log['episodes'].append(episode)
        V_approx, log_TD_0 = TD_0(self.env, pi, self.alpha, self.num_episodes)

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
            self.log['theta'].append(wrap_pi(self.env.x[0]))
            self.log['thetadot'].append(self.env.x[1])
            done = False
            while not done:
                a = np.argmax(pi[sp])
                (sp, rp, done) = self.env.step(a)
                self.log['t'].append(self.log['t'][-1] + 1)
                self.log['s'].append(sp)
                self.log['a'].append(a)
                self.log['r'].append(rp)
                self.log['theta'].append(wrap_pi(self.env.x[0]))
                self.log['thetadot'].append(self.env.x[1])
            
            plt.figure()
            plt.plot(self.log['t'], self.log['s'])
            plt.plot(self.log['t'][:-1], self.log['a'])
            plt.plot(self.log['t'][:-1], self.log['r'])
            plt.title("State, Action and Reward for Q-Learning")
            plt.xlabel("Time")
            plt.legend(['s', 'a', 'r'])
            plt.savefig('figures/pendulum/trajectories_qlearning.png')

            plt.figure()
            plt.plot(self.log['t'], self.log['theta'])
            plt.plot(self.log['t'], self.log['thetadot'])
            plt.axhline(y=np.pi, color='r', linestyle='-')
            plt.axhline(y=-np.pi, color='r', linestyle='-')
            plt.title(f"Theta, Theta_dot vs Time for Q-Learning")
            plt.xlabel("Time")
            plt.legend(['theta','theta_dot'])
            plt.savefig('figures/pendulum/theta_thetadot_qlearning.png')

            plt.figure()
            plt.plot(self.log['episodes'],self.log['G'])
            plt.xlabel("Number of Episodes")
            plt.ylabel("Total Return (G)")
            plt.title("Learning curve for number of iterations: Q-Learning")
            plt.savefig('figures/pendulum/learning_curve_qlearning.png')

            plt.figure()
            plt.plot(V_approx)
            plt.xlabel("States")
            plt.ylabel("Value Function")
            plt.title("State-Value Function Learned by TD(0): Q-Learning")
            plt.savefig('figures/pendulum/state_value_qlearning.png')

            plt.figure()
            plt.plot(log_TD_0['t'],log_TD_0['theta'])
            plt.plot(log_TD_0['t'],log_TD_0['thetadot'])
            plt.title("Theta, Theta_dot vs Time for TD(0) : Q-Learning")
            plt.xlabel("Time")
            plt.legend(['theta','theta_dot'])
            plt.savefig('figures/pendulum/theta_thetadot_qlearning_TD_0.png')
        return Q, pi, self.log


def main():
    env = Pendulum(n_theta=15, n_thetadot=21)
    env.reset()

    sarsa = SARSA(env,num_episodes=3000)
    Q_sarsa,pi_sarsa,log_sarsa = sarsa.SARSA(verbose=True, plots=True)

    qlearning = QLearning(env,num_episodes=3000)
    Q_qlearning,pi_qlearning,log_qlearning = qlearning.Q_Learning(verbose=True, plots=True)

    # plt.show()
    # Q_sarsa,pi_sarsa,log_sarsa = SARSA()
    # Q_qlearning,pi_qlearning,log_qlearning = Q_Learning()
    # log_list = [log_sarsa,log_qlearning]
    # pi_list = [pi_sarsa,pi_qlearning]
    # algorithms = {0:"SARSA",1:"Q-Learning"}
    # for i in range(len(log_list)):
    #     sp = env.reset()
    #     log_list[i]['s'].append(sp)
    #     log_list[i]['theta'].append(env.x[0])
    #     log_list[i]['thetadot'].append(env.x[1])
    #     done = False
    #     while not done:
    #         a = np.argmax(pi_list[i][sp])
    #         (sp, rp, done) = env.step(a)
    #         log_list[i]['t'].append(log_list[i]['t'][-1] + 1)
    #         log_list[i]['s'].append(sp)
    #         log_list[i]['a'].append(a)
    #         log_list[i]['r'].append(rp)
    #         log_list[i]['theta'].append(env.x[0])
    #         log_list[i]['thetadot'].append(env.x[1])
    #     plt.figure()
    #     plt.plot(log_list[i]['t'], log_list[i]['s'])
    #     plt.plot(log_list[i]['t'][:-1], log_list[i]['a'])
    #     plt.plot(log_list[i]['t'][:-1], log_list[i]['r'])
    #     plt.title(f"State, Action and Reward vs Time for {algorithms[i]}")
    #     plt.xlabel("Time")
    #     plt.legend(['s', 'a', 'r'])

    #     plt.figure()
    #     plt.plot(log_list[i]['t'], log_list[i]['theta'])
    #     plt.plot(log_list[i]['t'], log_list[i]['thetadot'])
    #     plt.title(f"Theta, Theta_dot vs Time for {algorithms[i]}")
    #     plt.xlabel("Time")
    #     plt.legend(['theta','theta_dot'])

    #     plt.figure()
    #     plt.plot(log_list[i]['episodes'],log_list[i]['G'])
    #     plt.xlabel("Number of episodes")
    #     plt.ylabel("Total return (G)")
    #     plt.title(f"Learning curve for number of episodes: {algorithms[i]}")
    
    # for i in range(2):
    #     if i == 0:
    #         alpha_vals = np.linspace(0,1,11)
    #         plt.figure()
    #         for alpha in alpha_vals:
    #             Q_alpha, pi_alpha, log_alpha = SARSA(alpha=alpha,verbose=False)
    #             plt.scatter(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
    #             plt.title(f"Learning curve for different alpha: {algorithms[i]}")
    #         plt.legend()

    #         epsilon_vals = np.linspace(0,0.5,11)
    #         plt.figure()
    #         for epsilon in epsilon_vals:
    #             Q_eps, pi_eps, log_eps = SARSA(epsilon=epsilon,verbose=False)
    #             plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
    #             plt.title(f"Learning curve for different epsilon: {algorithms[i]}")
    #         plt.legend()
        
    #     else:
    #         alpha_vals = np.linspace(0,1,11)
    #         plt.figure()
    #         for alpha in alpha_vals:
    #             Q_alpha, pi_alpha, log_alpha = Q_Learning(alpha=alpha,verbose=False)
    #             plt.scatter(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
    #             plt.title(f"Learning curve for different alpha: {algorithms[i]}")
    #         plt.legend()

    #         epsilon_vals = np.linspace(0,0.5,11)
    #         plt.figure()
    #         for epsilon in epsilon_vals:
    #             Q_eps, pi_eps, log_eps = Q_Learning(epsilon=epsilon,verbose=False)
    #             plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
    #             plt.title(f"Learning curve for different epsilon: {algorithms[i]}")
    #         plt.legend()

if __name__ == '__main__':
    main()