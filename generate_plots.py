"""
Reinforcement Learning: Q-Learning with Linear Function Approximation
Comprehensive Visualization Suite for 4x4 Gridworld

Author: Chief AI Officer, Google
"""

import os
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────
# Environment & Approximator Classes (self-contained)
# ─────────────────────────────────────────────────────────────


class GridWorld:
    def __init__(self):
        self.currentState = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'), (0, 1): ('L', 'D', 'R'), (0, 2): ('L', 'D', 'R'), (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'), (1, 1): ('U', 'L', 'D', 'R'), (1, 2): ('U', 'L', 'D', 'R'), (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'), (2, 1): ('U', 'L', 'D', 'R'), (2, 2): ('U', 'L', 'D', 'R'), (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'), (3, 1): ('U', 'L', 'R'), (3, 2): ('U', 'L', 'R')
        }
        self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -2}
        self.explored = 0
        self.exploited = 0

    def getRandomPolicy(self):
        policy = {}
        for state in self.actions:
            policy[state] = np.random.choice(self.actions[state])
        return policy

    def getCurrentState(self):
        if not self.currentState:
            self.currentState = (0, 0)
        return self.currentState

    def is_terminal(self, s):
        return s not in self.actions

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        row, col = state
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            col -= 1
        elif action == 'R':
            col += 1
        reward = self.rewards.get((row, col), 0)
        return action, (row, col), reward


class LinearApproximator:
    def __init__(self):
        self.theta = np.array([0.1, 0.1, 0.1, 0.1])
        self.theta_history = [self.theta.copy()]

    def state2Value(self, state):
        return ((state[0]-1) * self.theta[0] +
                (state[1]-1.5) * self.theta[1] +
                (state[0]*state[1]-3) * self.theta[2] +
                self.theta[3])

    def applyGD(self, state, target, learningrate=0.01):
        prediction = self.state2Value(state)
        error = target - prediction
        self.theta[0] += learningrate * error * state[0]
        self.theta[1] += learningrate * error * state[1]
        self.theta[2] += learningrate * error * (state[0] * state[1])
        self.theta[3] += learningrate * error
        self.theta_history.append(self.theta.copy())
        return error


# ─────────────────────────────────────────────────────────────
# Training Loop with Metrics Collection
# ─────────────────────────────────────────────────────────────

def train(num_episodes=2000, explore_rate=0.02, gamma=0.9):
    env = GridWorld()
    approx = LinearApproximator()
    policy = env.getRandomPolicy()

    episode_rewards = []
    episode_steps = []
    td_errors = []
    value_snapshots = []
    snapshot_episodes = []
    cumulative_rewards = []
    running_reward = 0

    for ep in range(num_episodes):
        state = env.getCurrentState()
        step = 0
        total_reward = 0
        ep_td_errors = []

        while not env.is_terminal(state) and step < 30:
            action, next_state, reward = env.move(state, policy, explore_rate)
            total_reward += reward
            state = next_state
            step += 1

            if env.is_terminal(next_state):
                target = reward
            else:
                target = reward + gamma * approx.state2Value(next_state)

            err = approx.applyGD(state, target)
            ep_td_errors.append(abs(err))

        episode_rewards.append(total_reward)
        episode_steps.append(step)
        td_errors.append(np.mean(ep_td_errors) if ep_td_errors else 0)
        running_reward += total_reward
        cumulative_rewards.append(running_reward / (ep + 1))

        if ep % 100 == 0 or ep == num_episodes - 1:
            snap = {}
            for s in env.actions:
                snap[s] = approx.state2Value(s)
            value_snapshots.append(snap)
            snapshot_episodes.append(ep)

    return (env, approx, policy, episode_rewards, episode_steps,
            td_errors, value_snapshots, snapshot_episodes, cumulative_rewards)


# ─────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────

COLORS = {
    'bg': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'text': '#c9d1d9',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'orange': '#d29922',
    'purple': '#bc8cff',
}


def style_ax(ax, title=''):
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=COLORS['text'],
                     fontsize=12, fontweight='bold', pad=10)


def plot_gridworld(ax, env, approx):
    """Plot the gridworld environment with learned values."""
    grid = np.zeros((4, 4))
    for state in env.actions:
        grid[state[0], state[1]] = approx.state2Value(state)
    grid[3, 3] = 5.0  # terminal

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(
        vmin=grid.min(), vcenter=0, vmax=max(grid.max(), 0.1))

    ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

    labels = {(0, 0): 'START', (3, 3): 'GOAL', (1, 3)              : 'HOLE', (2, 1): 'HOLE', (3, 1): 'HOLE'}
    for i in range(4):
        for j in range(4):
            label = labels.get((i, j), '')
            val = grid[i, j]
            color = 'white' if abs(val) > 2 else COLORS['text']
            if label:
                ax.text(j, i - 0.15, label, ha='center', va='center', fontsize=7,
                        fontweight='bold', color=color)
                ax.text(j, i + 0.2, f'{val:.2f}', ha='center',
                        va='center', fontsize=8, color=color)
            else:
                ax.text(j, i, f'{val:.2f}', ha='center',
                        va='center', fontsize=9, color=color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                       color=COLORS['text'], fontsize=8)
    ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'],
                       color=COLORS['text'], fontsize=8)
    ax.grid(True, color=COLORS['border'], linewidth=0.5)
    ax.set_title('Learned State-Value Landscape',
                 color=COLORS['text'], fontsize=12, fontweight='bold', pad=10)


def plot_optimal_path(ax, env, approx):
    """Plot the gridworld with derived optimal policy arrows."""
    grid = np.zeros((4, 4))
    for state in env.actions:
        grid[state[0], state[1]] = approx.state2Value(state)
    grid[3, 3] = 5.0

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(
        vmin=grid.min(), vcenter=0, vmax=max(grid.max(), 0.1))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

    arrow_map = {'U': (0, -0.35), 'D': (0, 0.35),
                 'L': (-0.35, 0), 'R': (0.35, 0)}

    for state in env.actions:
        best_action = None
        best_value = -np.inf
        for action in env.actions[state]:
            row, col = state
            if action == 'U':
                row -= 1
            elif action == 'D':
                row += 1
            elif action == 'L':
                col -= 1
            elif action == 'R':
                col += 1
            val = approx.state2Value((row, col)) if (
                row, col) in env.actions else (5.0 if (row, col) == (3, 3) else 0)
            if val > best_value:
                best_value = val
                best_action = action

        dx, dy = arrow_map[best_action]
        ax.annotate('', xy=(state[1] + dx, state[0] + dy),
                    xytext=(state[1], state[0]),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

    labels = {(0, 0): 'S', (3, 3): 'G', (1, 3): '×', (2, 1): '×', (3, 1): '×'}
    for (i, j), lbl in labels.items():
        color = COLORS['green'] if lbl == 'G' else (
            COLORS['red'] if lbl == '×' else COLORS['accent'])
        ax.text(j, i, lbl, ha='center', va='center', fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color=COLORS['border'], linewidth=0.5)
    ax.set_title('Derived Optimal Policy (Greedy)',
                 color=COLORS['text'], fontsize=12, fontweight='bold', pad=10)


def plot1_training_dashboard(env, approx, episode_rewards, episode_steps, td_errors, cumulative_rewards):
    """Figure 1: Training dynamics dashboard."""
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
    fig.suptitle('Training Dynamics Dashboard — Q-Learning with Linear FA',
                 color=COLORS['accent'], fontsize=16, fontweight='bold', y=0.97)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Episodic Reward
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Episode Reward')
    window = 50
    smoothed = np.convolve(
        episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(episode_rewards, alpha=0.15,
             color=COLORS['accent'], linewidth=0.5)
    ax1.plot(range(window-1, len(episode_rewards)), smoothed,
             color=COLORS['accent'], linewidth=2, label=f'{window}-ep moving avg')
    ax1.set_xlabel('Episode', color=COLORS['text'])
    ax1.set_ylabel('Total Reward', color=COLORS['text'])
    ax1.legend(
        facecolor=COLORS['card'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    # Steps per Episode
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Steps per Episode')
    smoothed_steps = np.convolve(
        episode_steps, np.ones(window)/window, mode='valid')
    ax2.plot(episode_steps, alpha=0.15, color=COLORS['green'], linewidth=0.5)
    ax2.plot(range(window-1, len(episode_steps)), smoothed_steps,
             color=COLORS['green'], linewidth=2, label=f'{window}-ep moving avg')
    ax2.set_xlabel('Episode', color=COLORS['text'])
    ax2.set_ylabel('Steps', color=COLORS['text'])
    ax2.legend(
        facecolor=COLORS['card'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    # TD Error
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Mean |TD Error| per Episode')
    smoothed_td = np.convolve(td_errors, np.ones(window)/window, mode='valid')
    ax3.plot(td_errors, alpha=0.15, color=COLORS['orange'], linewidth=0.5)
    ax3.plot(range(window-1, len(td_errors)), smoothed_td,
             color=COLORS['orange'], linewidth=2, label=f'{window}-ep moving avg')
    ax3.set_xlabel('Episode', color=COLORS['text'])
    ax3.set_ylabel('|TD Error|', color=COLORS['text'])
    ax3.legend(
        facecolor=COLORS['card'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    # Cumulative Average Reward
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'Cumulative Average Reward')
    ax4.plot(cumulative_rewards, color=COLORS['purple'], linewidth=2)
    ax4.axhline(y=cumulative_rewards[-1], color=COLORS['red'], linestyle='--', alpha=0.7,
                label=f'Final avg: {cumulative_rewards[-1]:.3f}')
    ax4.set_xlabel('Episode', color=COLORS['text'])
    ax4.set_ylabel('Avg Reward', color=COLORS['text'])
    ax4.legend(
        facecolor=COLORS['card'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    fig.savefig('assets/training_dashboard.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] training_dashboard.png')


def plot2_value_landscape(env, approx):
    """Figure 2: State-value landscape + optimal policy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])
    fig.suptitle('Learned Value Function & Optimal Policy',
                 color=COLORS['accent'], fontsize=16, fontweight='bold', y=0.98)
    for ax in axes:
        ax.set_facecolor(COLORS['card'])

    plot_gridworld(axes[0], env, approx)
    plot_optimal_path(axes[1], env, approx)

    fig.savefig('assets/value_policy_map.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] value_policy_map.png')


def plot3_theta_evolution(approx):
    """Figure 3: Weight parameter convergence."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['bg'])
    style_ax(ax, 'Linear Approximator Weight Convergence (θ₀ … θ₃)')

    history = np.array(approx.theta_history)
    labels = ['θ₀  (row feature)', 'θ₁  (col feature)',
              'θ₂  (interaction)', 'θ₃  (bias)']
    colors = [COLORS['accent'], COLORS['green'],
              COLORS['orange'], COLORS['purple']]

    for i in range(4):
        ax.plot(history[:, i], color=colors[i],
                linewidth=1.5, label=labels[i], alpha=0.85)

    ax.set_xlabel('Gradient Update Step', color=COLORS['text'])
    ax.set_ylabel('Weight Value', color=COLORS['text'])
    ax.legend(facecolor=COLORS['card'], edgecolor=COLORS['border'],
              labelcolor=COLORS['text'], fontsize=9)
    ax.axhline(0, color=COLORS['border'], linewidth=0.5, linestyle='--')

    fig.savefig('assets/theta_convergence.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] theta_convergence.png')


def plot4_value_evolution(value_snapshots, snapshot_episodes, env):
    """Figure 4: Value function evolution over training."""
    n = len(snapshot_episodes)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(
        4 * cols, 4 * rows), facecolor=COLORS['bg'])
    fig.suptitle('Value Function Evolution Across Training',
                 color=COLORS['accent'], fontsize=16, fontweight='bold', y=1.0)

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (snap, ep) in enumerate(zip(value_snapshots, snapshot_episodes)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_facecolor(COLORS['card'])

        grid = np.zeros((4, 4))
        for state in env.actions:
            grid[state[0], state[1]] = snap[state]
        grid[3, 3] = 5.0

        cmap = plt.cm.RdYlGn
        vmin, vmax = -3, 6
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(grid[i, j]) > 2 else COLORS['text'])
        ax.set_title(f'Episode {ep}', color=COLORS['text'], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(len(snapshot_episodes), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig('assets/value_evolution.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] value_evolution.png')


def plot5_exploration_exploitation(env):
    """Figure 5: Exploration vs exploitation pie chart."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['card'])

    sizes = [env.explored, env.exploited]
    labels = [f'Exploration\n({env.explored:,})',
              f'Exploitation\n({env.exploited:,})']
    colors_pie = [COLORS['orange'], COLORS['accent']]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie, explode=explode,
        autopct='%1.1f%%', startangle=140, textprops={'color': COLORS['text'], 'fontsize': 11})
    for at in autotexts:
        at.set_fontweight('bold')

    ax.set_title('Exploration vs Exploitation Ratio',
                 color=COLORS['text'], fontsize=13, fontweight='bold')

    fig.savefig('assets/explore_exploit.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] explore_exploit.png')


def plot6_architecture_diagram():
    """Figure 6: System architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('System Architecture: Q-Learning with Linear Function Approximation',
                 color=COLORS['accent'], fontsize=14, fontweight='bold', pad=20)

    def draw_box(x, y, w, h, label, sublabel='', color=COLORS['accent']):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                       facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                    fontsize=8, color=COLORS['text'], style='italic')

    def draw_arrow(x1, y1, x2, y2, label='', color=COLORS['text']):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.2, label, ha='center', va='bottom',
                    fontsize=8, color=COLORS['text'], style='italic')

    # Boxes
    draw_box(0.5, 2.5, 2.5, 2, 'GridWorld\nEnvironment',
             's, r, done', COLORS['green'])
    draw_box(4.5, 2.5, 2.5, 2, 'ε-Greedy\nAgent', 'π(s) → a', COLORS['accent'])
    draw_box(8.5, 2.5, 2.5, 2, 'Linear FA',
             'V̂(s;θ) = θᵀφ(s)', COLORS['purple'])
    draw_box(8.5, 0.2, 2.5, 1.5, 'SGD\nUpdater',
             'θ ← θ + α·δ·∇θ', COLORS['orange'])
    draw_box(4.5, 5, 2.5, 1.2, 'TD Target', 'r + γ·V̂(s\')', COLORS['red'])

    # Arrows
    draw_arrow(3, 3.5, 4.5, 3.5, 'state s')
    draw_arrow(7, 3.5, 8.5, 3.5, 'V̂(s)')
    draw_arrow(5.75, 2.5, 5.75, 1.8)
    ax.text(6.2, 2.0, 'action a', fontsize=8,
            color=COLORS['text'], style='italic')
    draw_arrow(5.75, 1.5, 1.75, 1.5)
    draw_arrow(1.75, 2.5, 1.75, 1.5)
    ax.text(3.5, 1.2, 's\', r', fontsize=8,
            color=COLORS['text'], style='italic')
    draw_arrow(9.75, 2.5, 9.75, 1.7, 'δ = target − V̂')
    draw_arrow(8.5, 1.0, 7.5, 1.0)
    draw_arrow(7.5, 1.0, 7.5, 2.8)
    ax.text(7.0, 1.8, 'θ update', fontsize=8,
            color=COLORS['text'], style='italic')
    draw_arrow(5.75, 5.0, 5.75, 4.5)
    draw_arrow(5.75, 6.2, 9.75, 4.5)
    ax.text(8.5, 5.6, 'V̂(s\')', fontsize=8,
            color=COLORS['text'], style='italic')

    fig.savefig('assets/architecture.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] architecture.png')


def plot7_feature_analysis(approx, env):
    """Figure 7: Feature contribution analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLORS['bg'])
    fig.suptitle('Feature Space & Contribution Analysis',
                 color=COLORS['accent'], fontsize=14, fontweight='bold', y=1.02)

    theta = approx.theta

    # Feature heatmaps
    features = [
        ('φ₁: Row Feature\n(row − 1)', lambda r, c: (r - 1)),
        ('φ₂: Col Feature\n(col − 1.5)', lambda r, c: (c - 1.5)),
        ('φ₃: Interaction\n(row×col − 3)', lambda r, c: (r * c - 3)),
    ]

    for idx, (title, fn) in enumerate(features):
        ax = axes[idx]
        ax.set_facecolor(COLORS['card'])
        grid = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                grid[i, j] = fn(i, j) * theta[idx]

        im = ax.imshow(grid, cmap='coolwarm', aspect='equal')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if abs(grid[i, j]) > 1 else COLORS['text'])
        ax.set_title(
            f'{title}\n(×θ{idx}={theta[idx]:.4f})', color=COLORS['text'], fontsize=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig('assets/feature_analysis.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] feature_analysis.png')


# ─────────────────────────────────────────────────────────────
# Multi-run Comparison
# ─────────────────────────────────────────────────────────────

def plot8_multi_run_comparison():
    """Figure 8: Statistical robustness across multiple runs."""
    n_runs = 20
    n_episodes = 2000
    all_rewards = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        env = GridWorld()
        approx = LinearApproximator()
        policy = env.getRandomPolicy()
        explore_rate = 0.02

        for ep in range(n_episodes):
            state = env.getCurrentState()
            step = 0
            total_reward = 0
            while not env.is_terminal(state) and step < 30:
                action, next_state, reward = env.move(
                    state, policy, explore_rate)
                total_reward += reward
                state = next_state
                step += 1
                target = reward if env.is_terminal(
                    next_state) else reward + 0.9 * approx.state2Value(next_state)
                approx.applyGD(state, target)
            all_rewards[run, ep] = total_reward

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['bg'])
    style_ax(ax, f'Statistical Robustness ({n_runs} Independent Runs)')

    window = 50
    smoothed = np.apply_along_axis(lambda x: np.convolve(
        x, np.ones(window)/window, mode='valid'), 1, all_rewards)

    mean_r = smoothed.mean(axis=0)
    std_r = smoothed.std(axis=0)
    x = np.arange(window - 1, n_episodes)

    ax.fill_between(x, mean_r - 2*std_r, mean_r + 2*std_r,
                    alpha=0.15, color=COLORS['accent'], label='±2σ')
    ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                    alpha=0.3, color=COLORS['accent'], label='±1σ')
    ax.plot(x, mean_r, color=COLORS['accent'],
            linewidth=2, label='Mean reward')

    ax.set_xlabel('Episode', color=COLORS['text'])
    ax.set_ylabel('Smoothed Reward', color=COLORS['text'])
    ax.legend(facecolor=COLORS['card'],
              edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    fig.savefig('assets/multi_run_robustness.png', dpi=180,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print('[✓] multi_run_robustness.png')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    np.random.seed(42)

    print('Training agent...')
    (env, approx, policy, episode_rewards, episode_steps,
     td_errors, value_snapshots, snapshot_episodes, cumulative_rewards) = train(num_episodes=2000)

    print(f'\nFinal θ = {approx.theta}')
    print(f'Explored: {env.explored:,}  |  Exploited: {env.exploited:,}')
    print(f'\nGenerating plots...\n')

    plot1_training_dashboard(env, approx, episode_rewards,
                             episode_steps, td_errors, cumulative_rewards)
    plot2_value_landscape(env, approx)
    plot3_theta_evolution(approx)
    plot4_value_evolution(value_snapshots, snapshot_episodes, env)
    plot5_exploration_exploitation(env)
    plot6_architecture_diagram()
    plot7_feature_analysis(approx, env)
    plot8_multi_run_comparison()

    print('\n✅  All plots saved to assets/')
