import gymnasium as gym
import numpy as np
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from typing import List, Tuple, Dict
from copy import copy
from gymnasium.spaces import Discrete, Box
import torch
from datetime import datetime
import os




def get_area_spell_coordinates(center, radius, grid_size):
    """
    Given a: 
    center: position on the board (normally the position of the initial enemy)
    radius: the area of the attack/spell
    grid_size: the size of the board
    
    It calculates the positions that are affected by a circular area with said radius, having said point of origin on a square board.
    """
    x_center, y_center = center
    width, height = grid_size
    affected_coords = []
    
    for x in range(x_center - radius, x_center + radius + 1):
        for y in range(y_center - radius, y_center + radius + 1):
            # Check if coordinates are within grid bounds
            if 0 <= x < width and 0 <= y < height:
                distance = abs(x - x_center) + abs(y - y_center)
                if distance <= radius:
                    affected_coords.append((x, y))
    
    return affected_coords



def bresenham_line(x1, y1, x2, y2):
    """
    Given the x and y positions of a character and the x and y positions on a board, 
    calculates which positions are between those two positions if a straight line were drawn based on Bresenhan's algorithm. 
    This function will be used to check if there is a wall or an obstacle between character 1 and character 2.
    """
    points = []
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy
    return points





class Job:
    """
    This class manages the character classes. These classes have the following attributes:    
    name: Class name
    level: The character's class level
    proficiency_bonus: The class's proficiency bonus. 
    spellcasting_ability = A value indicating which character's proficiency is used to cast spells in this job.
    """
    def __init__(self,
                 name: str,
                 level: int = 1,
                 proficiency_bonus: int = 0,
                 spellcasting_ability = str,
                 ):
        self.name = name
        self.level = level
        self.proficiency_bonus = proficiency_bonus
        self.spellcasting_ability = spellcasting_ability













# Falta meter en attack y en Character la mecanica de bufos
class Attack:
    """
    This class manages the characters' attacks. Attacks range from the most basic attacks, such as weapon attacks, 
    to individual and area-of-effect offensive spells, such as buffs and debuffs, to characters, such as healing spells. The attributes are as follows:
    
    name: The name of the attack, spell, etc.
    die_roll: Die roll is a dictionary where keys are the die type (e.g., 4, 6, 8, etc.) and values are the number of dice to roll. For example, {1: 4, 2: 6} means roll 1d4 and 2d6.
    number_turns_preparation: Number of turns required to prepare the attack, e.g., 0 for instant attacks, 1 for one turn preparation, etc.
    superior_level_buff: The die roll that will added to the damage if one spell slot of higher level is used.
    bonus_attr: Bonus attribute is a string indicating which attribute is added in the attack, e.g., 'strength', 'dexterity', etc.
    resource: Resource is a string indicating the type of resource used, e.g., 'spell slots', 'actions', etc.
    element: Element is a string indicating the type of element the attack uses, e.g., 'fire', 'ice', 'lightning', etc.
    saving_throw: Saving throw is a string indicating the type of saving throw required, e.g., 'dexterity', 'constitution', etc.
    concentration: Concentration is a boolean indicating if the attack requires concentration
    range: Range is an integer representing the range of the attack in tiles. For example, 1 means adjacent tile, 2 means two tiles away, etc.
    objetive: Objetive can be 'self', 'adventurer', 'enemy', 'area'.
    aoe: AOE represents the area of the attack
    self_status_effects: This attribute represents the buffs that the character gives himself when using the attack
    status_effects: Status refers to the combat effects like graple, prone, poisoned, etc. It is a dictionary where keys are the status effect names and values are tuples with the saving throw tipe the enemy need to do to avoid it and the value he needs to get.
    """
    def __init__(self,
                 name: str,
                 die_roll: Dict[int, int],
                 number_turns_preparation: int,
                 superior_level_buff: Dict[int, int],
                 bonus_attr: str,
                 resource: List[str],
                 element: str,
                 saving_throw: str,
                 concentration: bool,
                 range: int,
                 objetive: str,
                 aoe: int,
                 self_status_effects: Dict[str, Tuple[str, int]],
                 status_effects: Dict[str, Tuple[str, int]]):
       
        self.name = name
        self.number_turns_preparation = number_turns_preparation
        self.die_roll = die_roll
        self.superior_level_buff = superior_level_buff
        self.bonus_attr = bonus_attr
        self.resource = resource
        self.element = element
        self.saving_throw = saving_throw
        self.concentration = concentration
        self.range = range
        self.objetive = objetive
        self.aoe = aoe
        self.self_status_effects = self_status_effects
        self.status_effects = status_effects




# This class represents a character in the game, including their attributes, attacks, and resources.
class Character:
    """
    This role represents the characters that will compete in the environment and that the agents will control. This class has the following characteristics:
    name: The name of the character
    role: Role can be 'tank', 'healer', 'damage dealer', etc. It is not the class, is the role in their gameplay. It doesn't affect to the code
    level: Level is an integer representing the character's level.
    job: Job is the class of the character, like 'warrior', 'mage', etc. 
    tag: Tag is a string that can be used to identify the character in the game, like 'adventurer', 'enemy', etc.
    max_hp: The maximum hit points of the character
    hp: The current hit points of the character
    status: This atribute represents the status effects that the character has
    resistances: This atribute represents the resistances to certain attacks that the character has
    strength: The strength atribute of the character
    dexterity: The dexterity atribute of the character
    constitution: The constitution atribute of the character
    intelligence: The intelligence atribute of the character
    wisdom: The wisdom atribute of the character
    charisma: The charisma atribute of the character
    armor_class: The armor class atribute of the character
    position: The current position atribute of the character
    original_position: The original position atribute of the character that in the enviroment reset will respawn
    attacks: This atribute contains a number of attacks that the character has and will use in the combat
    resources: This atribute contains the resources of the character such as the movement points, the spells slots, the actions, etc
    targets: This list contains the posible targets that the character has in sight and can attack
    initiative: this parameter controls when the character will act in the turn cycle
    """
    def __init__(self,
                 name: str,
                 role: str,
                 level: int,
                 job: Job,
                 tag: str,
                 max_hp: int,
                 hp: int,
                 status: Dict[str, int],
                 resistances: Dict[str, int],
                 strength: int,
                 dexterity: int,
                 constitution: int,
                 intelligence: int,
                 wisdom: int,
                 charisma: int,
                 armor_class: int,
                 position: Tuple[int, int],
                 original_position: Tuple[int, int],
                 attacks: List[Attack],
                 resources: Dict[str, int],
                 targets: List[str],
                 initiative: int = 0
                ):
       
        self.name = name
        self.role = role
        self.job = job
        self.tag = tag
        self.level = level
        self.resistances = resistances
        self.strength = strength
        self.dexterity = dexterity
        self.constitution = constitution
        self.intelligence = intelligence
        self.wisdom = wisdom
        self.charisma = charisma
        self.armor_class = armor_class
        self.position = position
        self.original_position = original_position
        self.attacks = attacks
        self.resources = resources
        self.max_hp = max_hp
        self.hp = hp
        self.strength_bonus = (self.strength - 10) // 2
        self.dexterity_bonus = (self.dexterity - 10) // 2
        self.constitution_bonus = (self.constitution - 10) // 2
        self.intelligence_bonus = (self.intelligence - 10) // 2
        self.wisdom_bonus = (self.wisdom - 10) // 2
        self.charisma_bonus = (self.charisma - 10) // 2
        self.original_resources = resources.copy()
        self.original_movement_points = self.resources.get("movement_points", 0)
        self.original_action_slots = self.resources.get("action_slots", 0)
        self.original_bonus_action_slots = self.resources.get("bonus_action_slots", 0)
        self.targets = []
        self.status = status


    
    def see_resources_of_attack(self, attack: Attack) -> bool:
        """
        This function returns true if the character has enought resources to make an attack that needs some resources.
        For example a fireball need a spell_slots and an action so it will search if the character have still those resources and if so, it will return true
        """
        for resource in attack.resource:
            if resource not in self.resources or self.resources[resource] <= 0:
                return False
        return True
           


    def use_resource(self, resource: str, amount: int = 1) -> bool:
        """
        This function will use the resource in a certain amount
        """
        if resource in self.resources and self.resources[resource] >= amount:
            self.resources[resource] -= amount
            return True
        return False
















# Create a variable that conteins a list of points that represent the walls of the map




class DnDEnvironment(MultiAgentEnv):
    """
    This is the DND enviroment where all agents will be training to improve and learn to play D&D in a certain board
    This enviroment have some important functions related to the enviroment training that are:

    1. __init__():
    Initializes the game environment with configuration parameters
    Sets up characters with stats, abilities, positions, and resources
    Creates map with walls and character positions
    Defines observation and action spaces for RL agents

    2. _update_map():
    Helper function that updates the internal map representation
    Called whenever character positions change

    3. reset():
    Resets the environment to initial state
    Rolls character initiative based on stats and conditions to decide the new turn order
    Restores character HP, resources, and positions
    Starts new turn order

    4. render():
    Displays current game state in human-readable format
    Shows turn info, character statuses, and map visualization

    5. make_opportunity_attack():
    Handles attacks of opportunity when characters move away from an enemy that is near him
    Calculates attack rolls with advantage/disadvantage and applies damage based on character stats and conditions to the character that has move away

    6. step():
    Main game D&D loop that processes agent actions
    Handles movement, attacks, and anything that the character can do
    Manages turn order and resources of the character when made an action
    Calculates rewards obtained between the turn and checks winning conditions and their rewards
    Updates game state and returns observations
    """

    def __init__(self, config=None, max_turns=3000, map_size_x=5, map_size_y=5, walls = [] ,characters=[
        Character(
                name="player1",
                role="Human Fighter (Hammer)",
                level=1,
                job=Job(name="Human Fighter", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="adventurer",
                max_hp=40,
                hp=40,
                status={},
                resistances={},
                strength=18,
                dexterity=16,
                constitution=16,
                intelligence=8,
                wisdom=8,
                charisma=8,
                armor_class=15,
                position=(0, 0),
                original_position=(0, 0),
                attacks=[
                    Attack(
                        name="Maul",
                        die_roll={2: 6},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="strength",
                        resource=["action_slots"],
                        element="bludgeoning",
                        saving_throw="dexterity",
                        concentration=False,
                        range=1,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={"prone": ("constitution", 13)}
                    ),
                    Attack(
                        name="Dash",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"resources": ("movement_points", 30)},
                        status_effects={}
                    ),
                    Attack(
                        name="Hide",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"hide ": ("dexterity", 10)},
                        status_effects={}
                    ),
                    Attack(
                        name="Disengage",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"disengage ": ("disengage", 1)},
                        status_effects={}
                    )

                   
                   
                ],
                resources={"movement_points": 30, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            ),
            Character(
                name="player2",
                role="Wood Elf Rogue (Scout)",
                level=1,
                job=Job(name="Wood Elf Rogue", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="adventurer",
                max_hp=32,
                hp=32,
                status={},
                resistances={},
                strength=12,
                dexterity=16,
                constitution=14,
                intelligence=8,
                wisdom=14,
                charisma=8,
                armor_class=15,
                position=(0, 4),
                original_position=(0, 4),
                attacks=[
                    Attack(
                        name="Crossbow, light",
                        die_roll={1: 8},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="dexterity",
                        resource=["action_slots"],
                        element="piercing",
                        saving_throw="none",
                        concentration=False,
                        range=12,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    ),
                    Attack(
                        name="Dagger",
                        die_roll={1: 4},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="dexterity",
                        resource=["action_slots"],
                        element="slashing",
                        saving_throw="none",
                        concentration=False,
                        range=1,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    ),
                    Attack(
                        name="Dash",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"resources": ("movement_points", 35)},
                        status_effects={}
                    ),
                    Attack(
                        name="Hide",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"hide ": ("dexterity", 10)},
                        status_effects={}
                    ),
                    Attack(
                        name="Disengage",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"disengage ": ("disengage", 1)},
                        status_effects={}
                    )
                ],
                resources={"movement_points": 35, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            ),
            Character(
                name="player3",
                role="Awakened Tree",
                level=1,
                job=Job(name="Awakened Tree", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="enemy",
                max_hp=59,
                hp=59,
                status={},
                resistances={"bludgeoning":0.5, "piercing":0.5},
                strength=19,
                dexterity=6,
                constitution=15,
                intelligence=10,
                wisdom=10,
                charisma=7,
                armor_class=13,
                position=(0, 3),
                original_position=(0, 3),
                attacks=[
                    Attack(
                        name="Slam",
                        die_roll={2: 8},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="strength",
                        resource=["action_slots"],
                        element="bludgeoning",
                        saving_throw="none",
                        concentration=False,
                        range=2,
                        objetive="adventurer",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    )
                ],
                resources={"movement_points": 20, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            )
        ]
    ):
        super().__init__()
        self.max_turns = max_turns
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.characters = characters if characters is not None else []
        self.current_turn = 0
        self.walls = walls if walls is not None else []
        self.copy_characters = copy(self.characters)
        self.num_enemies = sum(1 for char in self.characters if char.tag == 'enemy' and char.hp > 0)
        self.num_adventurers = sum(1 for char in self.characters if char.tag == 'adventurer' and char.hp > 0)
        # The number of agents will be one for each character
        self.agents = self.possible_agents = [char.name for char in self.characters]
        # Because of how PPO works it is necessary to "flatten" the map from a 2D matrix to a single dimensional map
        self.observation_spaces = {
            char.name: Box(0.0, 3.0, (self.map_size_x * self.map_size_y,), dtype=np.float32) for char in self.characters
        }
        # Each character will have a number of action spaces equal to the 4 possible lateral movements, the end of turn action, the number of attacks they have, and the total number of characters in the environment.
        self.action_spaces = {
            char.name: Discrete(5+ len(char.attacks) + self.num_enemies + self.num_adventurers) for char in self.characters
        }
        # The initialization of the current character and map is null in init, this is handled by the reset function
        self.current_player = None
        self.current_agent_mask = np.zeros(9, "float32")
        self.new_next_turn = True
        self.map = np.zeros(self.map_size_x * self.map_size_y, dtype=np.float32)
        self._update_map()


    def _update_map(self):
        """Helper method to update the flattened map representation"""
        self.map = np.zeros(self.map_size_x * self.map_size_y, dtype=np.float32)
        for char in self.characters:
            pos = char.position[1] * self.map_size_y + char.position[0]
            if char.tag == 'adventurer' and char.hp > 0:
                self.map[pos] = 1
            elif char.tag == 'enemy' and char.hp > 0:
                self.map[pos] = 2
        for wall in self.walls:
            pos = wall[1] * self.map_size_y + wall[0]
            self.map[pos] = 3


    def reset(self, *, seed=None, options=None):
        self.current_turn = 0
        for char in self.characters:
            # Check for certain states that may affect the result of the dice roll for the initiative
            current_is_incapacitated  = hasattr(char, 'status') and 'incapacitated' in char.status and char.status['incapacitated'] == 1
            current_is_invisible = hasattr(char, 'status') and 'invisible' in char.status and char.status['invisible'] == 1
            current_is_paralyzed = hasattr(char, 'status') and 'paralyzed' in char.status and char.status['paralyzed'] == 1
            current_is_petrified = hasattr(char, 'status') and 'petrified' in char.status and char.status['petrified'] == 1
            current_is_stuned = hasattr(char, 'status') and 'stunned' in char.status and char.status['stunned'] == 1
            current_is_unconscious = hasattr(char, 'status') and 'unconscious' in char.status and char.status['unconscious'] == 1
            # If the character has disadvantage on initiative and does not have advantage on initiative then the formula of a D20 + the dexterity bonus + 5 is used
            if (current_is_incapacitated or current_is_paralyzed or current_is_petrified or current_is_stuned or current_is_unconscious) and not (current_is_invisible):
                char.initiative = np.random.randint(1, 21) + char.dexterity_bonus + 5
            # If the character has advantage on initiative and does not have disadvantage on initiative then the formula of a D20 + the dexterity bonus + 15 is used
            elif not (current_is_incapacitated or current_is_paralyzed or current_is_petrified or current_is_stuned or current_is_unconscious) and (current_is_invisible):
                char.initiative = np.random.randint(1, 21) + char.dexterity_bonus + 15
            # If the character has neither advantage nor disadvantage on initiative then the formula of a D20 + the dexterity bonus + 10 is used
            else:
                char.initiative = np.random.randint(1, 21) + char.dexterity_bonus + 10
        # The characters are ordered in order of initiative.
        self.characters.sort(key=lambda char: char.initiative, reverse=True)
        # Values such as resources, health, targets and the original position of the characters are reset.
        for char in self.characters:
            char.resources = char.original_resources.copy()
            char.hp = char.max_hp
            char.targets = []
            char.position = char.original_position
        # Certain attributes are reset
        self.waiting_for_target = False  
        self.current_attack = None
        # The map is updated as the character positions are now updated.
        self._update_map()
        # The current character is established by the one who is first in the initiative
        self.current_player = self.characters[0].name

        return {
            self.current_player: np.array(self.map, dtype=np.float32),
        }, {}



    def _log_battle_results(self, winner):
        """Helper method to log battle results to a text file"""
        # Create the filename
        filename = "battle_results.txt"
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(filename)
        
        # Open the file in append mode
        with open(filename, 'a') as file:
            # Write header if file doesn't exist
            if not file_exists:
                file.write("BATTLE RESULTS LOG\n")
                file.write("=" * 50 + "\n")
            
            # Write current date and time
            file.write(f"\nBattle on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("-" * 30 + "\n")
            
            # Write character HP information
            file.write("CHARACTER HP STATUS:\n")
            for char in self.characters:
                file.write(f"{char.name} ({char.tag}): {char.hp}/{char.max_hp} HP\n")
            
            # Write the winner
            file.write(f"\nWINNING TEAM: {winner}\n")
            file.write("=" * 50 + "\n")



    def render(self, mode='human'):
        if mode == 'human':
            # Clear the console
            print("\033c", end="")
            # Print turn information
            print(f"Turn: {self.current_turn}")
            print(f"Current Player: {self.current_player}")
            print("-" * 40)  
            # Print character statuses
            print("Character Status:")
            for char in self.characters:
                status_str = ", ".join([f"{k}:{v}" for k, v in char.status.items()]) if char.status else "None"
                print(f"{char.name} ({char.tag}): HP {char.hp}/{char.max_hp} | Position {char.position} | Status: {status_str}")   
            print("-" * 40)
            # Print the map
            print("D&D Map:")
            for y in range(self.map_size_y):
                row = []
                for x in range(self.map_size_x):
                    pos = x * self.map_size_y + y
                    if self.map[pos] == 1:  # Adventurer
                        char = next((c for c in self.characters if c.position == (x, y) and c.tag == 'adventurer'), None)
                        row.append(f"A{char.name[-1]}" if char else "A?")
                    elif self.map[pos] == 2:  # Enemy
                        char = next((c for c in self.characters if c.position == (x, y) and c.tag == 'enemy'), None)
                        row.append(f"E{char.name[-1]}" if char else "E?")
                    elif self.map[pos] == 3:  # Wall
                        row.append(" W ")
                    else:
                        row.append(" . ")
                print(" ".join(row))
            print("-" * 40)
            # Print last action
            if hasattr(self, 'last_action'):
                current_char = next((c for c in self.characters if c.name == self.current_player), None)
                if current_char:
                    if self.last_action < 5:
                        action_names = ["Move Down", "Move Right", "Move Up", "Move Left", "Pass"]
                        print(f"Last Action: {action_names[self.last_action]}")
                    elif 5 <= self.last_action < 5 + len(current_char.attacks):
                        attack = current_char.attacks[self.last_action - 5]
                        print(f"Last Action: Used {attack.name}")
                    else:
                        target_idx = self.last_action - (5 + len(current_char.attacks))
                        if target_idx < len(self.characters):
                            target = self.characters[target_idx]
                            attack = getattr(self, 'current_attack', None)
                            if attack:
                                print(f"Last Action: Attacked {target.name} with {attack.name}")
            print("=" * 40)
            
        else:
            super().render(mode=mode)



    def make_opportunity_attack(self, enemy, current_character):
        # Use the first attack (that will normaly be your melee attack)
        attack = enemy.attacks[0]
        # Check some status from the enemy and you that have impact in the calculation of this opportunity attack
        has_advantage = hasattr(enemy, 'status') and 'advantage' in enemy.status and enemy.status['advantage'] == 1
        enemy_has_disadvantage = hasattr(current_character, 'status') and 'disadvantage' in current_character.status and current_character.status['disadvantage'] == 1
        has_disadvantage = hasattr(enemy, 'status') and 'disadvantage' in enemy.status and enemy.status['disadvantage'] == 1
        current_is_blinded = hasattr(enemy, 'status') and 'blinded' in enemy.status and enemy.status['blinded'] == 1
        enemy_is_blinded = hasattr(current_character, 'status') and 'blinded' in current_character.status and current_character.status['blinded'] == 1
        current_is_exhausted = hasattr(enemy, 'status') and 'exhausted' in enemy.status and enemy.status['exhausted'] == 1
        current_is_frightened = hasattr(enemy, 'status') and 'frightened' in enemy.status and enemy.status['frightened'] == 1
        current_is_grappled = hasattr(enemy, 'status') and 'grapple' in enemy.status and enemy.status['grapple'] == 1
        enemy_is_invisible = hasattr(current_character, 'status') and 'invisible' in current_character.status and current_character.status['invisible'] == 1
        current_is_invisible = hasattr(enemy, 'status') and 'invisible' in enemy.status and enemy.status['invisible'] == 1
        enemy_is_paralyzed = hasattr(current_character, 'status') and 'paralyzed' in current_character.status and current_character.status['paralyzed'] == 1
        enemy_is_petrified = hasattr(current_character, 'status') and 'petrified' in current_character.status and current_character.status['petrified'] == 1
        current_is_poisoned = hasattr(enemy, 'status') and 'poisoned' in enemy.status and enemy.status['poisoned'] == 1
        enemy_is_proned = hasattr(current_character, 'status') and 'prone' in current_character.status and current_character.status['prone'] == 1
        current_is_proned = hasattr(enemy, 'status') and 'prone' in enemy.status and enemy.status['prone'] == 1
        enemy_is_restrained = hasattr(current_character, 'status') and 'restrained' in current_character.status and current_character.status['restrained'] == 1
        current_is_restrained = hasattr(enemy, 'status') and 'restrained' in enemy.status and enemy.status['restrained'] == 1
        enemy_is_stuned = hasattr(current_character, 'status') and 'stunned' in current_character.status and current_character.status['stunned'] == 1
        enemy_is_unconscious = hasattr(current_character, 'status') and 'unconscious' in current_character.status and current_character.status['unconscious'] == 1
        enemy_is_hiding = hasattr(current_character, 'status') and 'hide' in current_character.status and current_character.status['hide'] == 1
        enemy_is_dodging = hasattr(current_character, 'status') and 'dodging' in current_character.status and current_character.status['dodging'] == 1


        # Roll attack with advantage/disadvantage if applicable (the status effects like prone will be check in these lines)
        if (has_advantage or enemy_has_disadvantage or enemy_is_blinded or current_is_invisible or enemy_is_paralyzed or enemy_is_petrified or enemy_is_proned or enemy_is_restrained or enemy_is_stuned or enemy_is_unconscious) and not (has_disadvantage or current_is_blinded or current_is_exhausted or current_is_frightened or current_is_grappled or enemy_is_invisible or current_is_poisoned or current_is_proned or current_is_restrained or enemy_is_hiding or (enemy_is_dodging and current_character.resources["movement_points"] > 0)):
            # Roll twice and take the higher result
            roll1 = np.random.randint(1, 21)
            roll2 = np.random.randint(1, 21)
            attack_roll = max(roll1, roll2)
        elif not (has_advantage or enemy_has_disadvantage or enemy_is_blinded or current_is_invisible or enemy_is_paralyzed or enemy_is_petrified or enemy_is_proned or enemy_is_restrained or enemy_is_stuned or enemy_is_unconscious) and (has_disadvantage or current_is_blinded or current_is_exhausted or current_is_frightened or current_is_grappled or enemy_is_invisible or current_is_poisoned or current_is_proned or current_is_restrained or enemy_is_hiding or (enemy_is_dodging and current_character.resources["movement_points"] > 0)):
            # Roll twice and take the lower result
            roll1 = np.random.randint(1, 21)
            roll2 = np.random.randint(1, 21)
            attack_roll = min(roll1, roll2)
        else:
            # Normal roll
            attack_roll = np.random.randint(1, 21)
                    
        # Get the appropriate attack bonus based on the attack's bonus attribute
        attack_bonus = 0          
        if attack.bonus_attr == "strength":
            attack_bonus = enemy.strength_bonus
        elif attack.bonus_attr == "dexterity":
            attack_bonus = enemy.dexterity_bonus
        elif attack.bonus_attr == "constitution":
            attack_bonus = enemy.constitution_bonus
        elif attack.bonus_attr == "intelligence":
            attack_bonus = enemy.intelligence_bonus
        elif attack.bonus_attr == "wisdom":
            attack_bonus = enemy.wisdom_bonus
        elif attack.bonus_attr == "charisma":
            attack_bonus = enemy.charisma_bonus

        # Add the greatest proficiency_bonus among all jobs the character has
        if hasattr(enemy, "job"):
            jobs = enemy.job
            # Jobs can be a list or a single Job
            if isinstance(jobs, list):
                max_prof = max((job.proficiency_bonus for job in jobs if hasattr(job, "proficiency_bonus")), default=0)
            else:
                max_prof = getattr(jobs, "proficiency_bonus", 0)
            attack_bonus += max_prof

        # First set the damage to 0 
        total_damage = 0
        # Now is time to check if the attack has passed the armor class of your character, and if that attack roll of early isnt a crit roll (if is a 20) or a crit failure (if is a 1)
        if current_character.hp > 0 and (attack_roll + attack_bonus) >= current_character.armor_class and attack_roll != 1 and attack_roll != 20:
            # Calculate the damage based on the die roll
            for count, die in attack.die_roll.items():
                for _ in range(count):
                    damage = np.random.randint(1, die + 1)
                    total_damage += damage
                    
            # Add the same attribute bonus to damage
            if attack.bonus_attr == "strength":
                total_damage += enemy.strength_bonus
            elif attack.bonus_attr == "dexterity":
                total_damage += enemy.dexterity_bonus
            elif attack.bonus_attr == "constitution":
                total_damage += enemy.constitution_bonus
            elif attack.bonus_attr == "intelligence":
                total_damage += enemy.intelligence_bonus
            elif attack.bonus_attr == "wisdom":
                total_damage += enemy.wisdom_bonus
            elif attack.bonus_attr == "charisma":
                total_damage += enemy.charisma_bonus
            # See if you are weak or have resistance to the type of that attack
            if attack.element:
                resistance_multiplier = current_character.resistances.get(attack.element, 1)
                total_damage = int(total_damage * resistance_multiplier)


        # If the attack roll is a 1 the attack won't do damage
        elif current_character.hp > 0 and attack_roll == 1:
            total_damage == 0


        # If the attacks roll is a 20 then it will crit and twice dice rolls will be rolled. The rest remains the same
        elif current_character.hp > 0 and attack_roll == 20:
            # Calculate the damage based on the die roll
            for count, die in attack.die_roll.items():
                for _ in range(count):
                    damage = np.random.randint(1, die + 1)
                    total_damage += damage
                    damage = np.random.randint(1, die + 1)
                    total_damage += damage
                
            # Add the same attribute bonus to damage
            if attack.bonus_attr == "strength":
                total_damage += enemy.strength_bonus
            elif attack.bonus_attr == "dexterity":
                total_damage += enemy.dexterity_bonus
            elif attack.bonus_attr == "constitution":
                total_damage += enemy.constitution_bonus
            elif attack.bonus_attr == "intelligence":
                total_damage += enemy.intelligence_bonus
            elif attack.bonus_attr == "wisdom":
                total_damage += enemy.wisdom_bonus
            elif attack.bonus_attr == "charisma":
                total_damage += enemy.charisma_bonus
            
            # See if you are weak or have resistance to the type of that attack
            if attack.element:
                resistance_multiplier = current_character.resistances.get(attack.element, 1)
                total_damage = int(total_damage * resistance_multiplier)           

        # Now the damage will be reduce the health
        current_character.hp = max(0, current_character.hp - total_damage)
        return total_damage






    def step(self, action_dict):
        # You see what action the agent whose turn it is has to take.
        action = action_dict[self.current_player]
        # You see what the index of the current character is
        current_index = next((i for i, char in enumerate(self.characters) if char.name == self.current_player), -1)
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {self.current_player: 0.0}
        # Create a terminateds-dict with the special `__all__` agent ID, indicating that if True, the episode ends for all agents.
        terminateds = {"__all__": False}

        # If the current agent is dead because his character HP is 0 then we skip their
        if self.characters[current_index].hp <= 0:
            self.current_player = self.characters[( [c.name for c in self.characters].index(self.current_player) + 1 ) % len(self.characters)].name
            if current_index == len(self.characters) - 1:
                self.current_turn += 1
            return (
                {self.current_player: np.array(self.map, dtype=np.float32)},
                rewards,
                terminateds,
                {},
                {},
            )
          
        # In steps, there are two phases: 
        # The first in which the character can move, end their turn, and perform attacks as long as those attacks have valid targets. 
        # The second phase, which occurs once an attack has been selected, is the only available action. 
        # In the first phase, target selection actions will be blocked by a mask, while in the second, the only actions that are not blocked are those that select valid targets.
        if self.waiting_for_target==False:
            self.current_agent_mask = np.zeros(5 + len(self.characters[current_index].attacks) + self.num_enemies + self.num_adventurers, "float32")
            self.current_agent_mask[0:5] = 1  # Movement actions are always available
            #action = actions[current_agent]
            # Check if the between the action 5 and the action 5 + len(self.characters[current_index].attacks) - 1 is a valid action if the distance between any character and the current character is less or equal to the range of the attack
            for i in range(len(self.characters[current_index].attacks)):
                attack = self.characters[current_index].attacks[i]
                choice_made = False
                if attack.objetive == 'enemy' or attack.objetive == 'anyone' or attack.objetive == 'area':
                    for char in self.characters:
                        if char.tag == 'enemy' and char.hp > 0 and choice_made == False:
                            distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                            if distance <= attack.range:
                                #self.observation_spaces[current_agent]["action_mask"][5 + i] = 1
                                # Using bresenham_line function to check if the path is clear
                                path_clear = True
                                for x, y in bresenham_line(self.characters[current_index].position[0], self.characters[current_index].position[1], char.position[0], char.position[1]):
                                    if self.map[y * self.map_size_y + x] == 3:  # Wall
                                        path_clear = False
                                if path_clear:
                                    self.current_agent_mask[5 + i] = 1
                                    choice_made = True
                                elif not path_clear:
                                    # If the path is not clear, the action is not valid
                                    #self.observation_spaces[current_agent]["action_mask"][5 + i] = 0
                                    self.current_agent_mask[5 + i] = 0
                            else:
                                #self.observation_spaces[current_agent]["action_mask"][5 + i] = 0
                                self.current_agent_mask[5 + i] = 0
                elif attack.objetive == 'adventurer' or attack.objetive == 'anyone' or attack.objetive == 'area':
                    for char in self.characters:
                        if char.tag == 'adventurer' and char.hp > 0 and choice_made == False:
                            distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                            if distance <= attack.range:
                                #self.observation_spaces[current_agent]["action_mask"][5 + i] = 1
                                path_clear = True
                                for x, y in bresenham_line(self.characters[current_index].position[0], self.characters[current_index].position[1], char.position[0], char.position[1]):
                                    if self.map[y * self.map_size_y + x] == 3:  # Wall
                                        path_clear = False
                                if path_clear:
                                    self.current_agent_mask[5 + i] = 1
                                    choice_made = True
                                elif not path_clear:
                                    # If the path is not clear, the action is not valid
                                    #self.observation_spaces[current_agent]["action_mask"][5 + i] = 0
                                    self.current_agent_mask[5 + i] = 0
                            else:
                                #self.observation_spaces[current_agent]["action_mask"][5 + i] = 0
                                self.current_agent_mask[5 + i] = 0
                elif attack.objetive == 'self' and self.current_player == self.characters[current_index].name:
                    # If the attack is a self target, then the action is always valid
                    #self.observation_spaces[current_agent]["action_mask"][5 + i] = 1
                    self.current_agent_mask[5 + i] = 1

        current_char = self.characters[current_index]
        current_pos = current_char.position[0] * self.map_size_y + current_char.position[1]

        # If the character use dodge in the previous turn then in this one he lose the dodging bonus
        current_is_dodging = hasattr(self.characters[current_index], 'status') and 'dodging' in self.characters[current_index].status and self.characters[current_index].status['dodging'] == 1
        if current_is_dodging and self.new_next_turn == True:
            current_char.status['dodging'] = 0
            self.new_next_turn = False

        # If the character is proned then it must spend half the speed to remove the prone condition
        current_is_proned = hasattr(self.characters[current_index], 'status') and 'prone' in self.characters[current_index].status and self.characters[current_index].status['prone'] == 1
        if current_is_proned:
            # Half the total movement points and put the status prone to 0
            current_char.resources["movement_points"] = current_char.resources["movement_points"] // 2
            current_char.status['prone'] = 0

        if action == 0:  # Move down
            new_y = current_char.position[1] + 1
            if (new_y < self.map_size_y and
                current_char.resources["movement_points"] >= 5 and
                self.map[current_char.position[0] * self.map_size_y + new_y] == 0 and
                self.current_agent_mask[action] == 1):
                original_position = current_char.position
                current_char.position = (current_char.position[0], new_y)
                current_char.resources["movement_points"] -= 5
                rewards[self.current_player] += 1


                for i, char in enumerate(self.characters):
                    if char.tag == 'enemy' and char.hp > 0 and current_char.tag == 'adventurer':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

                        
                    elif char.tag == 'adventurer' and char.hp > 0 and current_char.tag == 'enemy':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

            else:
                rewards[self.current_player] -= 0.5
       
        elif action == 1:  # Move right
            new_x = current_char.position[0] + 1
            if (new_x < self.map_size_x and
                current_char.resources["movement_points"] >= 5 and
                self.map[new_x * self.map_size_y + current_char.position[1]] == 0 and
                self.current_agent_mask[action] == 1):
                original_position = current_char.position
                current_char.position = (new_x, current_char.position[1])
                current_char.resources["movement_points"] -= 5
                rewards[self.current_player] += 1

                for i, char in enumerate(self.characters):
                    if char.tag == 'enemy' and char.hp > 0 and current_char.tag == 'adventurer':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

                        
                    elif char.tag == 'adventurer' and char.hp > 0 and current_char.tag == 'enemy':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

                
            else:
                rewards[self.current_player] -= 0.5
       
        elif action == 2:  # Move up
            new_y = current_char.position[1] - 1
            if (new_y >= 0 and
                current_char.resources["movement_points"] >= 5 and
                self.map[current_char.position[0] * self.map_size_y + new_y] == 0 and
                self.current_agent_mask[action] == 1):
                original_position = current_char.position
                current_char.position = (current_char.position[0], new_y)
                current_char.resources["movement_points"] -= 5
                rewards[self.current_player] += 1

                for i, char in enumerate(self.characters):
                    if char.tag == 'enemy' and char.hp > 0 and current_char.tag == 'adventurer':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

                        
                    elif char.tag == 'adventurer' and char.hp > 0 and current_char.tag == 'enemy':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

            else:
                rewards[self.current_player] -= 0.5
       
        elif action == 3:  # Move left
            new_x = current_char.position[0] - 1
            if (new_x >= 0 and
                current_char.resources["movement_points"] >= 5 and
                self.map[new_x * self.map_size_y + current_char.position[1]] == 0 and
                self.current_agent_mask[action] == 1):
                original_position = current_char.position
                current_char.position = (new_x, current_char.position[1])
                current_char.resources["movement_points"] -= 5
                rewards[self.current_player] += 1
                for i, char in enumerate(self.characters):
                    if char.tag == 'enemy' and char.hp > 0 and current_char.tag == 'adventurer':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5

                        
                    elif char.tag == 'adventurer' and char.hp > 0 and current_char.tag == 'enemy':
                        original_distance = abs(original_position[0] - char.position[0]) + abs(original_position[1] - char.position[1])
                        new_distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' in current_char.status and "reactions" in char.resources:
                            if current_char.status['disengage'] == 0 and char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5
                        elif new_distance > 1 and original_distance <= 1 and hasattr(current_char, 'status') and 'disengage' not in current_char.status and "reactions" in char.resources:
                            if char.resources["reactions"] > 0:
                                char.resources["reactions"] -= 1
                                #rewards[self.current_player] -= self.make_opportunity_attack(char, current_char)
                                self.make_opportunity_attack(char, current_char)
                                rewards[self.current_player] -= 0.5


            else:
                rewards[self.current_player] -= 0.5
       
        # The action 4 pass the turn to the next agent in the turn order
        elif action == 4 and self.current_agent_mask[action] == 1:
            # Pass turn to the next agent
            if "movement_points" in self.characters[current_index].resources and "movement_points" in self.characters[current_index].original_resources:
                self.characters[current_index].resources["movement_points"] = self.characters[current_index].original_resources["movement_points"]
            if "action_slots" in self.characters[current_index].resources and "action_slots" in self.characters[current_index].original_resources:
                self.characters[current_index].resources["action_slots"] = self.characters[current_index].original_resources["action_slots"]
            if "bonus_action_slots" in self.characters[current_index].resources and "bonus_action_slots" in self.characters[current_index].original_resources:
                self.characters[current_index].resources["bonus_action_slots"] = self.characters[current_index].original_resources["bonus_action_slots"]
            if "reactions" in self.characters[current_index].resources and "reactions" in self.characters[current_index].original_resources:
                self.characters[current_index].resources["reactions"] = self.characters[current_index].original_resources["reactions"]
            self.characters[current_index].targets = []
            self.current_attack = None
            self.new_next_turn = True
            self.current_player = self.characters[( [c.name for c in self.characters].index(self.current_player) + 1 ) % len(self.characters)].name
            if current_index == len(self.characters) - 1:
                self.current_turn += 1


        # If you are in a middle of attack and the end turn action is masked, then a negative reward is added
        elif action == 4 and self.current_agent_mask[action] == 0:
            rewards[self.current_player] -= 0.5


        # For each action between 5 and 5 + len(self.characters[current_index].attacks) - 1, use the attack of the character. If the character dont have the resource of that attack, the action should be invalid even in the selections
        elif 5 <= action < 5 + len(self.characters[current_index].attacks) and self.current_agent_mask[action] == 1:
            attack_index = action - 5
            attack_index = action - 5
            attack = self.characters[current_index].attacks[attack_index]
            # Check if the character has enough resources to use the attack
            if self.characters[current_index].see_resources_of_attack(attack): #Change this and put in the action mask condition
                # Use the resources of the attack
                for resource in attack.resource:
                    self.characters[current_index].use_resource(resource)
                # Apply the attack to the target
                self.current_attack = attack
                # Create a new variable list with size of the number of characters named adventurers and enemies that has 0 if the ataack range value is equal or less of the true distance between this agent and the other character
                self.targets = []
                for i, copy_char in enumerate(self.copy_characters):  # Use copy_characters for order
                    # Find the corresponding character in self.characters (same name or ID)
                    char = next((c for c in self.characters if c.name == copy_char.name), None)
                    if char is None:
                        continue
                    if char.tag == 'enemy' and char.hp > 0 and (self.current_attack.objetive == 'enemy' or self.current_attack.objetive == 'area' or self.current_attack.objetive == 'anyone'):
                        distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if distance <= self.current_attack.range:
                            self.targets.append(1)
                            self.waiting_for_target = True
                        else:
                            self.targets.append(0)
                    elif char.tag == 'adventurer' and char.hp > 0 and (self.current_attack.objetive == 'adventurer' or self.current_attack.objetive == 'area' or self.current_attack.objetive == 'anyone'):
                        distance = abs(self.characters[current_index].position[0] - char.position[0]) + abs(self.characters[current_index].position[1] - char.position[1])
                        if distance <= self.current_attack.range:
                            self.targets.append(1)
                            self.waiting_for_target = True
                        else:
                            self.targets.append(0)
                    # If the target is a self target, then the target is the current character
                    elif self.current_attack.objetive == 'self' and char.name == self.current_player:
                        self.targets.append(1)
                        self.waiting_for_target = True
                    else:
                        self.targets.append(0)
                # Block all other actions using the action mask but the actions that are between 5 + len(self.characters[current_index].attacks) and 5 + len(self.characters[current_index].attacks) + self.num_enemies + self.num_adventurers
                for i in range(5 + len(self.characters[current_index].attacks), 5 + len(self.characters[current_index].attacks) + self.num_enemies + self.num_adventurers):
                    # If the variable self.targets in the position i - (5 + len(self.characters[current_index].attacks)) is equal to 1, then the action mask in that position should be 1, otherwise it should be 0
                    if self.targets[i - (5 + len(self.characters[current_index].attacks))] == 1:
                        self.current_agent_mask[i] = 1
                    else:
                        self.current_agent_mask[i] = 0
                for i in range(0, 5 + len(self.characters[current_index].attacks)):
                    self.current_agent_mask[i] = 0



        elif 5 + len(self.characters[current_index].attacks) <= action < 5 + len(self.characters[current_index].attacks) + self.num_enemies + self.num_adventurers and self.current_agent_mask[action] == 1:
            # This action is to switch the target of the attack, so we need to check if the character has targets and if the action is valid
            target_index = action - (5 + len(self.characters[current_index].attacks))
            target_name = self.copy_characters[target_index].name
            # Find the corresponding character in self.characters (updated data)
            target_enemy = next((c for c in self.characters if c.name == target_name), None)

            # Check if the attack gives an advantage or dissadvantage before the attaks was made
            if hasattr(self.current_attack, 'self_status_effects'):
                for effect_name, (effect_type, effect_value) in self.current_attack.self_status_effects.items():
                    if effect_name == "advantage" or effect_name == "disadvantage":
                        # Add status effect to current character
                        if not hasattr(self.characters[current_index], 'status'):
                            self.characters[current_index].status = {}
                        self.characters[current_index].status[effect_type] = effect_value
                    elif effect_name == "resources":
                        # Add resource to current character
                        if not hasattr(self.characters[current_index], 'resources'):
                            self.characters[current_index].resources = {}
                        if effect_type not in self.characters[current_index].resources:
                            self.characters[current_index].resources[effect_type] = effect_value
                        else:
                            # If it exists, add the effect_value to the current value
                            if (self.current_attack.name == "Dash"):
                                self.characters[current_index].resources["movement_points"] += self.characters[current_index].resources["movement_points"]
                            else:
                                self.characters[current_index].resources[effect_type] += effect_value
            
            
            # If the attack is to only one target (meaning the area of the attack will only affect to the targeted character) then this code will run
            if hasattr(self.current_attack, 'aoe') and self.current_attack.aoe == 0:
                # Check some status from the enemy and you that have impact in the calculation of this opportunity attack
                has_advantage = hasattr(self.characters[current_index], 'status') and 'advantage' in self.characters[current_index].status and self.characters[current_index].status['advantage'] == 1
                enemy_has_disadvantage = hasattr(target_enemy, 'status') and 'disadvantage' in target_enemy.status and target_enemy.status['disadvantage'] == 1
                has_disadvantage = hasattr(self.characters[current_index], 'status') and 'disadvantage' in self.characters[current_index].status and self.characters[current_index].status['disadvantage'] == 1
                current_is_blinded = hasattr(self.characters[current_index], 'status') and 'blinded' in self.characters[current_index].status and self.characters[current_index].status['blinded'] == 1
                enemy_is_blinded = hasattr(target_enemy, 'status') and 'blinded' in target_enemy.status and target_enemy.status['blinded'] == 1
                current_is_exhausted = hasattr(self.characters[current_index], 'status') and 'exhausted' in self.characters[current_index].status and self.characters[current_index].status['exhausted'] == 1
                current_is_frightened = hasattr(self.characters[current_index], 'status') and 'frightened' in self.characters[current_index].status and self.characters[current_index].status['frightened'] == 1
                current_is_grappled = hasattr(self.characters[current_index], 'status') and 'grapple' in self.characters[current_index].status and self.characters[current_index].status['grapple'] == 1
                enemy_is_invisible = hasattr(target_enemy, 'status') and 'invisible' in target_enemy.status and target_enemy.status['invisible'] == 1
                current_is_invisible = hasattr(self.characters[current_index], 'status') and 'invisible' in self.characters[current_index].status and self.characters[current_index].status['invisible'] == 1
                enemy_is_paralyzed = hasattr(target_enemy, 'status') and 'paralyzed' in target_enemy.status and target_enemy.status['paralyzed'] == 1
                enemy_is_petrified = hasattr(target_enemy, 'status') and 'petrified' in target_enemy.status and target_enemy.status['petrified'] == 1
                current_is_poisoned = hasattr(self.characters[current_index], 'status') and 'poisoned' in self.characters[current_index].status and self.characters[current_index].status['poisoned'] == 1
                enemy_is_proned = hasattr(target_enemy, 'status') and 'prone' in target_enemy.status and target_enemy.status['prone'] == 1
                current_is_proned = hasattr(self.characters[current_index], 'status') and 'prone' in self.characters[current_index].status and self.characters[current_index].status['prone'] == 1
                enemy_is_restrained = hasattr(target_enemy, 'status') and 'restrained' in target_enemy.status and target_enemy.status['restrained'] == 1
                current_is_restrained = hasattr(self.characters[current_index], 'status') and 'restrained' in self.characters[current_index].status and self.characters[current_index].status['restrained'] == 1
                enemy_is_stuned = hasattr(target_enemy, 'status') and 'stunned' in target_enemy.status and target_enemy.status['stunned'] == 1
                enemy_is_unconscious = hasattr(target_enemy, 'status') and 'unconscious' in target_enemy.status and target_enemy.status['unconscious'] == 1
                enemy_is_hiding = hasattr(target_enemy, 'status') and 'hide' in target_enemy.status and target_enemy.status['hide'] == 1
                enemy_is_dodging = hasattr(target_enemy, 'status') and 'dodging' in target_enemy.status and target_enemy.status['dodging'] == 1
                # Roll attack with advantage/disadvantage if applicable
                if (has_advantage or enemy_has_disadvantage or enemy_is_blinded or current_is_invisible or enemy_is_paralyzed or enemy_is_petrified or enemy_is_proned or enemy_is_restrained or enemy_is_stuned or enemy_is_unconscious) and not (has_disadvantage or current_is_blinded or current_is_exhausted or current_is_frightened or current_is_grappled or enemy_is_invisible or current_is_poisoned or current_is_proned or current_is_restrained or enemy_is_hiding or (enemy_is_dodging and target_enemy.resources["movement_points"] > 0)):
                    # Roll twice and take the higher result
                    roll1 = np.random.randint(1, 21)
                    roll2 = np.random.randint(1, 21)
                    attack_roll = max(roll1, roll2)
                elif not (has_advantage or enemy_has_disadvantage or enemy_is_blinded or current_is_invisible or enemy_is_paralyzed or enemy_is_petrified or enemy_is_proned or enemy_is_restrained or enemy_is_stuned or enemy_is_unconscious) and (has_disadvantage or current_is_blinded or current_is_exhausted or current_is_frightened or current_is_grappled or enemy_is_invisible or current_is_poisoned or current_is_proned or current_is_restrained or enemy_is_hiding or (enemy_is_dodging and target_enemy.resources["movement_points"] > 0)):
                    # Roll twice and take the lower result
                    roll1 = np.random.randint(1, 21)
                    roll2 = np.random.randint(1, 21)
                    attack_roll = min(roll1, roll2)
                else:
                    # Normal roll
                    attack_roll = np.random.randint(1, 21)
                
                # Get the appropriate attack bonus based on the attack's bonus attribute
                attack_bonus = 0
                if self.current_attack.bonus_attr == "strength":
                    attack_bonus = self.characters[current_index].strength_bonus
                elif self.current_attack.bonus_attr == "dexterity":
                    attack_bonus = self.characters[current_index].dexterity_bonus
                elif self.current_attack.bonus_attr == "constitution":
                    attack_bonus = self.characters[current_index].constitution_bonus
                elif self.current_attack.bonus_attr == "intelligence":
                    attack_bonus = self.characters[current_index].intelligence_bonus
                elif self.current_attack.bonus_attr == "wisdom":
                    attack_bonus = self.characters[current_index].wisdom_bonus
                elif self.current_attack.bonus_attr == "charisma":
                    attack_bonus = self.characters[current_index].charisma_bonus

                # Add the greatest proficiency_bonus among all jobs the character has
                if hasattr(self.characters[current_index], "job"):
                    jobs = self.characters[current_index].job
                    # Jobs can be a list or a single Job
                    if isinstance(jobs, list):
                        max_prof = max((job.proficiency_bonus for job in jobs if hasattr(job, "proficiency_bonus")), default=0)
                    else:
                        max_prof = getattr(jobs, "proficiency_bonus", 0)
                    attack_bonus += max_prof

                # Now is time to calculate the total damage
                total_damage = 0
                if target_enemy.hp > 0 and (attack_roll + attack_bonus) >= target_enemy.armor_class and attack_roll != 1 and attack_roll != 20:
                    # Calculate the damage based on the die roll
                    for count, die in self.current_attack.die_roll.items():
                        for _ in range(count):
                            damage = np.random.randint(1, die + 1)
                            total_damage += damage
                
                    # Add the same attribute bonus to damage
                    if self.current_attack.bonus_attr == "strength":
                        total_damage += self.characters[current_index].strength_bonus
                    elif self.current_attack.bonus_attr == "dexterity":
                        total_damage += self.characters[current_index].dexterity_bonus
                    elif self.current_attack.bonus_attr == "constitution":
                        total_damage += self.characters[current_index].constitution_bonus
                    elif self.current_attack.bonus_attr == "intelligence":
                        total_damage += self.characters[current_index].intelligence_bonus
                    elif self.current_attack.bonus_attr == "wisdom":
                        total_damage += self.characters[current_index].wisdom_bonus
                    elif self.current_attack.bonus_attr == "charisma":
                        total_damage += self.characters[current_index].charisma_bonus


                    if self.current_attack.element:
                        resistance_multiplier = target_enemy.resistances.get(self.current_attack.element, 1)
                        total_damage = int(total_damage * resistance_multiplier)

                # If the attack roll was a 1 then it will miss always so no damage will be dealt
                elif target_enemy.hp > 0 and attack_roll == 1:
                    total_damage == 0

                # If the attack roll was a 20 then twice dices will be rolled
                elif target_enemy.hp > 0 and attack_roll == 20:
                    # Calculate the damage based on the die roll
                    for count, die in self.current_attack.die_roll.items():
                        for _ in range(count):
                            damage = np.random.randint(1, die + 1)
                            total_damage += damage
                            damage = np.random.randint(1, die + 1)
                            total_damage += damage
                
                    # Add the same attribute bonus to damage
                    if self.current_attack.bonus_attr == "strength":
                        total_damage += self.characters[current_index].strength_bonus
                    elif self.current_attack.bonus_attr == "dexterity":
                        total_damage += self.characters[current_index].dexterity_bonus
                    elif self.current_attack.bonus_attr == "constitution":
                        total_damage += self.characters[current_index].constitution_bonus
                    elif self.current_attack.bonus_attr == "intelligence":
                        total_damage += self.characters[current_index].intelligence_bonus
                    elif self.current_attack.bonus_attr == "wisdom":
                        total_damage += self.characters[current_index].wisdom_bonus
                    elif self.current_attack.bonus_attr == "charisma":
                        total_damage += self.characters[current_index].charisma_bonus


                    if self.current_attack.element:
                        resistance_multiplier = target_enemy.resistances.get(self.current_attack.element, 1)
                        total_damage = int(total_damage * resistance_multiplier)           

                # Finally the damage is dealt to the enemy
                target_enemy.hp = max(0, target_enemy.hp - total_damage)
                rewards[self.current_player] += total_damage


            # Search if the attack has an atribute called aoe and his value is greater than 0. If so, that means that is an area attack and this code will run instead of the single target one
            elif hasattr(self.current_attack, 'aoe') and self.current_attack.aoe > 0:
                # Get all characters in the AoE radius
                aoe_targets = get_area_spell_coordinates(
                    center=target_enemy.position,
                    radius=self.current_attack.aoe,
                    grid_size=(self.map_size_x, self.map_size_y)
                )

                # Damage all valid targets in the area
                for target_pos in aoe_targets:
                    # Find character at this position
                    for char in self.characters:
                        if hasattr(char, 'position') and char.position == target_pos and char.position != self.characters[current_index].position:
                            aoe_path_clear = True
                            for x, y in bresenham_line(target_enemy.position[0], target_enemy.position[1], char.position[0], char.position[1]):
                                if self.map[y * self.map_size_y + x] == 3:
                                    aoe_path_clear = False
                            if aoe_path_clear:

                                aoe_damage = 0
                                
                                # Get attack bonus (same as before)
                                if self.current_attack.saving_throw == "strength":
                                    saving_throw_bonus = char.strength_bonus
                                elif self.current_attack.saving_throw == "dexterity":
                                    saving_throw_bonus = char.dexterity_bonus
                                elif self.current_attack.saving_throw == "constitution":
                                    saving_throw_bonus = char.constitution_bonus
                                elif self.current_attack.saving_throw == "intelligence":
                                    saving_throw_bonus = char.intelligence_bonus
                                elif self.current_attack.saving_throw == "wisdom":
                                    saving_throw_bonus = char.wisdom_bonus
                                elif self.current_attack.saving_throw == "charisma":
                                    saving_throw_bonus = char.charisma_bonus
                                else:
                                    saving_throw_bonus = 0

                                save_roll = np.random.randint(1, 21) + saving_throw_bonus

                                # Check some status that affect to the calculation of the save roll
                                char_is_dodging = hasattr(char, 'status') and 'dodging' in char.status and char.status['dodging'] == 1
                                char_is_restrained = hasattr(char, 'status') and 'restrained' in char.status and char.status['restrained'] == 1
                                char_is_paralyzed = hasattr(char, 'status') and 'paralyzed' in char.status and char.status['paralyzed'] == 1
                                char_is_stunned = hasattr(char, 'status') and 'stunned' in char.status and char.status['stunned'] == 1
                                char_is_petrified = hasattr(char, 'status') and 'petrified' in char.status and char.status['petrified'] == 1
                                char_is_unconscious = hasattr(char, 'status') and 'unconscious' in char.status and char.status['unconscious'] == 1
                                if self.current_attack.saving_throw == "dexterity" and char_is_dodging:
                                    save_roll1 = np.random.randint(1, 21) + saving_throw_bonus
                                    save_roll2 = np.random.randint(1, 21) + saving_throw_bonus
                                    save_roll = max(save_roll1, save_roll2)
                                elif self.current_attack.saving_throw == "dexterity" and char_is_restrained:
                                    save_roll1 = np.random.randint(1, 21) + saving_throw_bonus
                                    save_roll2 = np.random.randint(1, 21) + saving_throw_bonus
                                    save_roll = min(save_roll1, save_roll2)
                                elif (self.current_attack.saving_throw == "dexterity" or self.current_attack.saving_throw == "strength") and (char_is_paralyzed or char_is_stunned or char_is_petrified or char_is_unconscious):
                                    save_roll = 1
                                save_throw = 8
                                # Add the greatest proficiency_bonus among all jobs the character has
                                if hasattr(self.characters[current_index], "job"):
                                    jobs = self.characters[current_index].job
                                    # jobs can be a list or a single Job
                                    if isinstance(jobs, list):
                                        max_prof_job = max(jobs, key=lambda job: getattr(job, "proficiency_bonus", 0))
                                    else:
                                        max_prof_job = jobs
                                        max_prof = getattr(max_prof_job, "proficiency_bonus", 0)
                                        save_throw += max_prof
                                        # Use the spellcasting_ability of the job with max proficiency_bonus
                                        spell_attr = getattr(max_prof_job, "spellcasting_ability", None)
                                        if spell_attr:
                                            bonus_attr_value = getattr(self.characters[current_index], f"{spell_attr}_bonus", 0)
                                            save_throw += bonus_attr_value

                                # If the enemy make the save throw then he only takes the half of the damage
                                if char.hp > 0 and save_roll >= save_throw:
                                    for count, die in self.current_attack.die_roll.items():
                                        for _ in range(count):
                                            aoe_damage += np.random.randint(1, die + 1)
                                    aoe_damage = aoe_damage // 2
                                
                                # If the enemy fails the save throw then he only takes the full damage
                                elif char.hp > 0 and save_roll < save_throw:
                                    for count, die in self.current_attack.die_roll.items():
                                        for _ in range(count):
                                            aoe_damage += np.random.randint(1, die + 1)

                                # Appends to the total damage the stat that the attack uses as a bonus of damage
                                if self.current_attack.bonus_attr == "strength":
                                    aoe_damage += self.characters[current_index].strength_bonus
                                elif self.current_attack.bonus_attr == "dexterity":
                                    aoe_damage += self.characters[current_index].dexterity_bonus
                                elif self.current_attack.bonus_attr == "constitution":
                                    aoe_damage += self.characters[current_index].constitution_bonus
                                elif self.current_attack.bonus_attr == "intelligence":
                                    aoe_damage += self.characters[current_index].intelligence_bonus
                                elif self.current_attack.bonus_attr == "wisdom":
                                    aoe_damage += self.characters[current_index].wisdom_bonus
                                elif self.current_attack.bonus_attr == "charisma":
                                    aoe_damage += self.characters[current_index].charisma_bonus

                                # In the final calculation it sees if the target has any elemental resustance or weakness of that attack
                                if self.current_attack.element:
                                    resistance_multiplier = char.resistances.get(self.current_attack.element, 1)
                                    aoe_damage = int(aoe_damage * resistance_multiplier) 
                                
                                # Now the damage substrats the targets life
                                char.hp = max(0, char.hp - aoe_damage)

                                # If you hit an ally then you will receive a negative reward and if you hit someone of the opposite team you will give a positive reward
                                if (char.tag == 'enemy' and self.characters[current_index].tag == "adventurer") or (char.tag == 'adventurer' and self.characters[current_index].tag == "enemy"):
                                    rewards[self.current_player] += aoe_damage
                                elif (char.tag == 'enemy' and self.characters[current_index].tag == "enemy") or (char.tag == 'adventurer' and self.characters[current_index].tag == "adventurer"):
                                    rewards[self.current_player] -= aoe_damage
                                            
            self.waiting_for_target = False


        else:
            rewards[self.current_player] -= 0.5

        # In each step the map need to be updated
        self._update_map()
        
        # If all the enemies are dead then the adventurers will win a positive reward and the enemies will have a negative reward
        if all(char.hp <= 0 for char in self.characters if char.tag == 'enemy'):
            for char in self.characters:
                if char.tag == 'adventurer' and char.name in rewards:
                    rewards[char.name] += 10
                elif char.tag == 'enemy' and char.name in rewards:
                    rewards[char.name] -= 5
            self.last_action = action
            self._log_battle_results("Adventurers")
            #self.render()
            terminateds["__all__"] = True


        # If all the adventurers are dead then the adventurers will win a negative reward and the enemies will have a positive reward
        elif all(char.hp <= 0 for char in self.characters if char.tag == 'adventurer'):
            for char in self.characters:
                if char.tag == 'adventurer' and char.name in rewards:
                    rewards[char.name] -= 5
                elif char.tag == 'enemy' and char.name in rewards:
                    rewards[char.name] += 10
            self.last_action = action
            self._log_battle_results("Enemies")
            #self.render()
            terminateds["__all__"] = True

        #If the current turn reach the established max turns then the trial will be stoped
        if self.current_turn >= self.max_turns:
            self.last_action = action
            #self.render()
            self._log_battle_results("Draw - Max Turns Reached")
            terminateds["__all__"] = True
           

        return (
            {self.current_player: np.array(self.map, dtype=np.float32)},
            rewards,
            terminateds,
            {},
            {},
        )
   










characcters=[
        Character(
                name="player1",
                role="Human Fighter (Hammer)",
                level=1,
                job=Job(name="Human Fighter", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="adventurer",
                max_hp=40,
                hp=40,
                status={},
                resistances={},
                strength=18,
                dexterity=16,
                constitution=16,
                intelligence=8,
                wisdom=8,
                charisma=8,
                armor_class=15,
                position=(0, 0),
                original_position=(0, 0),
                attacks=[
                    Attack(
                        name="Maul",
                        die_roll={2: 6},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="strength",
                        resource=["action_slots"],
                        element="bludgeoning",
                        saving_throw="dexterity",
                        concentration=False,
                        range=1,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={"prone": ("constitution", 13)}
                    ),
                    Attack(
                        name="Dash",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"resources": ("movement_points", 30)},
                        status_effects={}
                    ),
                    Attack(
                        name="Hide",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"hide ": ("dexterity", 10)},
                        status_effects={}
                    ),
                    Attack(
                        name="Disengage",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"disengage ": ("disengage", 1)},
                        status_effects={}
                    )

                   
                   
                ],
                resources={"movement_points": 30, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            ),
            Character(
                name="player2",
                role="Wood Elf Rogue (Scout)",
                level=1,
                job=Job(name="Wood Elf Rogue", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="adventurer",
                max_hp=32,
                hp=32,
                status={},
                resistances={},
                strength=12,
                dexterity=16,
                constitution=14,
                intelligence=8,
                wisdom=14,
                charisma=8,
                armor_class=15,
                position=(0, 4),
                original_position=(0, 4),
                attacks=[
                    Attack(
                        name="Crossbow, light",
                        die_roll={1: 8},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="dexterity",
                        resource=["action_slots"],
                        element="piercing",
                        saving_throw="none",
                        concentration=False,
                        range=12,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    ),
                    Attack(
                        name="Dagger",
                        die_roll={1: 4},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="dexterity",
                        resource=["action_slots"],
                        element="slashing",
                        saving_throw="none",
                        concentration=False,
                        range=1,
                        objetive="enemy",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    ),
                    Attack(
                        name="Dash",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"resources": ("movement_points", 35)},
                        status_effects={}
                    ),
                    Attack(
                        name="Hide",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"hide ": ("dexterity", 10)},
                        status_effects={}
                    ),
                    Attack(
                        name="Disengage",
                        die_roll={0: 0},
                        number_turns_preparation=0,
                        superior_level_buff={0: 0},
                        bonus_attr="none",
                        resource=["action_slots"],
                        element="none",
                        saving_throw="none",
                        concentration=False,
                        range=0,
                        objetive="self",
                        aoe=0,
                        self_status_effects={"disengage ": ("disengage", 1)},
                        status_effects={}
                    )
                ],
                resources={"movement_points": 35, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            ),
            Character(
                name="player3",
                role="Awakened Tree",
                level=1,
                job=Job(name="Awakened Tree", level=1, proficiency_bonus=2, spellcasting_ability = "wisdom"),
                tag="enemy",
                max_hp=59,
                hp=59,
                status={},
                resistances={"bludgeoning":0.5, "piercing":0.5},
                strength=19,
                dexterity=6,
                constitution=15,
                intelligence=10,
                wisdom=10,
                charisma=7,
                armor_class=13,
                position=(0, 3),
                original_position=(0, 3),
                attacks=[
                    Attack(
                        name="Slam",
                        die_roll={2: 8},
                        number_turns_preparation=0,
                        superior_level_buff={},
                        bonus_attr="strength",
                        resource=["action_slots"],
                        element="bludgeoning",
                        saving_throw="none",
                        concentration=False,
                        range=2,
                        objetive="adventurer",
                        aoe=0,
                        self_status_effects={},
                        status_effects={}
                    )
                ],
                resources={"movement_points": 30, "action_slots":1, "bonus_action_slots":1, "reactions": 1},
                targets=[]
            )
        ]





if __name__ == "__main__":
    parser = add_rllib_example_script_args(
        default_iters=300, default_timesteps=1200000, default_reward=700
    )
    parser.set_defaults(
        enable_new_api_stack=True,
        num_agents=3,
    )
    args = parser.parse_args()
    args.checkpoint_freq = 10
    # Coment this if you dont want to use a checkpoint
    #args.checkpoint_path = "C:\\Users\\jadlp\\OneDrive\\Documentos\\TFM\\PPO_2025-08-29_14-15-40\\PPO_DnDEnvironment_e0e7c_00000_0_2025-08-29_14-15-40\\checkpoint_000017"
    assert args.num_agents == 3, "Must set --num-agents= equal to the number of players when running this script!"


    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(DnDEnvironment,
                     env_config={
                        "max_turns":3000,
                        "map_size_x":5,
                        "map_size_y":5,
                        "walls": [],
                        "characters": characcters
                     })
        .multi_agent(
            policies={"player1", "player2", "player3"},
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
        )
        .env_runners(
            num_env_runners=11,  
        )
    )


    run_rllib_example_script_experiment(base_config, args)
