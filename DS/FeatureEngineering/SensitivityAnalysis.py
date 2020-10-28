# Morris Method
#%%

import numpy as np
from SALib.sample import saltelli
from SALib.sample import morris as smorris
from SALib.analyze import sobol
from SALib.analyze import morris as amorris
from SALib.test_functions import Ishigami


# Define the model inputs
problem = {'num_vars': 3, 'names': ['x1', 'x2', 'x3'], 'bounds': [[-3.14159265359, 3.14159265359], [-3.14159265359, 3.14159265359], [-3.14159265359, 3.14159265359]]}

# Generate samples
paramValues = saltelli.sample(problem, 1000)

# Run model (example)
Y = Ishigami.evaluate(paramValues)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
print(Si['S1'])


X = smorris.sample(problem, 1000, num_levels=4, grid_jump=2)
Y = Ishigami.evaluate(X)
Si = amorris.analyze(problem, X, Y, conf_level=0.95, print_to_console=True, num_levels=4, grid_jump=2)
print(Si)


# Tornado plot
#%%

import numpy as np
from matplotlib import pyplot as plt

def Tornado(variables, base, lows, values):
    # The y position for each variable
    ys = range(len(values))[::-1]  # top to bottom

    # Plot the bars, one by one
    for y, low, value in zip(ys, lows, values):
        # The width of the 'low' and 'high' pieces
        low_width = base - low
        high_width = low + value - base
        # Each bar is a "broken" horizontal bar chart
        plt.broken_barh([(low, low_width), (base, high_width)], (y - 0.4, 0.8), facecolors=['blue', 'green'], edgecolors=['black', 'black'], linewidth=1,)

        # Display the value as text. It should be positioned in the center of the 'high' bar, except if there isn't any room there, then it should be next to bar instead.
        x = base + high_width / 2
        if x <= base + 50: x = base + high_width + 50
        plt.text(x, y, str(value), va='center', ha='center')

    # Draw a vertical line down the middle
    plt.axvline(base, color='black')

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    axes = plt.gca()  # (gca = get current axes)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.xaxis.set_ticks_position('top')

    # Make the y-axis display the variables
    plt.yticks(ys, variables)

    # Set the portion of the x- and y-axes to show
    plt.xlim(base - 1000, base + 1000)
    plt.ylim(-1, len(variables))
    plt.show()

# Test
variables = ['apple', 'juice', 'orange', 'peach', 'gum', 'stones', 'bags', 'lamps',]
base = 3000
lows = np.array([base-246/2, base-1633/2, base-500/2, base-150/2, base-35/2, base-36/2, base-43/2, base-37/2,])
values = np.array([246, 1633, 500, 150, 35, 36, 43, 37,])
Tornado(variables, base, lows, values)
