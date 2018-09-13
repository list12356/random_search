from models.langford.mdp import Action
from models.langford.mdp import AntiShape
from models.langford.mdp import ComboLock
import random

num_states = 100

def run(m, num_epochs):
    # a trivial controller that takes random actions
    while (num_epochs > 0):
        a = Action.go_right
        if (random.random() < 0.5):
	        a = Action.go_left
        sf, reward = m.next_state(a)
        if (reward >= 0.):
            print("current state: "  + str(sf.state) + " reward = " + str(sf.reward))
            num_epochs = num_epochs - 1

def main():
    m = AntiShape()
    print("random controller on antishape")
    m.create(num_states)
    run(m, 20)

    print("random controller on combolock")
    m.create(num_states)
    run(m, 20)

if __name__ == "__main__":
    # execute only if run as a script
    main()