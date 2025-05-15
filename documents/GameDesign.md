Epic Game Mechanics and AI Systems
GameDesign/Mechanics/Epic

# Introduction

In this document we explore the interplay between advanced game mechanics, AI behavior, and loot systems in an epic fantasy MMO. Our goal is to outline the core principles, dive into technical details, and provide concrete examples suitable for retrieval-augmented generation testing.

# 1. Movement and Traversal

1.1. **Terrain Adaptation**  
Characters automatically adjust their movement speed and animations based on terrain type—forests, swamps, mountains, and deserts each impose unique modifiers.

1.2. **Climbing and Vaulting**  
Players can climb low walls or vault over obstacles when the surface angle is under 60 degrees. The engine performs a dynamic trace to detect adjacent “grabbable” ledges.

1.3. **Swimming and Diving**  
When entering water volumes:  
- A “swim” state replaces the “walk/run” animation  
- Stamina drains 1 point per 2 seconds underwater  
- Diving deeper than 10 meters triggers a “pressure check” that can cause damage over time

# 2. Combat and Abilities

2.1. **Combo System**  
Each class has a 3‑tier combo chain: light, medium, heavy attacks. Executing the third attack within 1 second of the previous one grants a “combo finish” bonus multiplier (1.2× damage).

2.2. **Cooldown and Resource Management**  
Abilities consume either Mana, Energy, or Rage. Cooldowns are global per-class but can be reduced by equipping certain artifacts. For example:  
- Artifact of Haste reduces all cooldowns by 5%  
- Ring of Fury increases Rage gain by 10%

2.3. **AI Target Prioritization**  
Enemy AI uses a scoring function: 
Score = (ThreatLevel × DistanceWeight) – (HealthPercentage × EvasionWeight)

- `ThreatLevel`: how much damage the player class can deal  
- `DistanceWeight`: inverse proportional to distance to target  
- `EvasionWeight`: higher for ranged attackers to bias AI toward melee targets

# 3. Navigation and Pathfinding

3.1. **NavMesh Generation**  
During level-building the editor voxelizes the environment at 0.5m resolution, then runs a flood-fill to merge connected navigable areas into convex polygons.

3.2. **Dynamic Obstacle Avoidance**  
AI agents run RVO2 (Reciprocal Velocity Obstacles) at 20Hz to locally avoid static and moving obstacles, while high‑level A* pathfinding occurs every 1s to replan around major blockers.

3.3. **Zone Tagging and Portal Links**  
Dungeon entrances and exits are marked as “portals.” When an agent reaches a portal boundary, the system queues a “zone change” event, unloading the old NavMesh and loading the new one asynchronously.

# 4. Loot and Reward Systems

4.1. **Weighted Drop Tables**  
Each monster type has a table of potential drops, each with a base weight. Loot is rolled via weighted random sampling, scaled by player luck attribute:

- Base weights:  
  - Common: 70  
  - Uncommon: 20  
  - Rare: 9  
  - Legendary: 1

- If player Luck > 100, multiply rare and legendary weights by 1.1 per 50 points of Luck.

4.2. **Pity and Soft‑Caps**  
A “pity counter” increments on each kill without a rare drop. At 50 kills, the next kill is guaranteed at least an Uncommon drop. After 200 kills, guaranteed Rare.

4.3. **Procedural Affix Generation**  
When an item drops, the system selects 1–3 affixes (prefixes/suffixes) from a list, using a secondary weight table that factors in item level and monster difficulty.

# 5. World Events and Persistence

5.1. **Global Event Scheduler**  
A server‑side cron schedules world events (e.g., “Dragon Siege”) which broadcast to all connected clients. A JSON payload defines event triggers, timers, and special loot rewards.

5.2. **Persistent State Management**  
Key world variables (e.g., boss health, portal status) are stored in a distributed NoSQL store. Agents subscribe to change feeds to update local AI behaviors in real time.

5.3. **Player‑Driven Economy**  
An auction house service aggregates buy and sell orders. Price ceilings/floors are enforced by a “market regulator” that adjusts fees dynamically to stabilize economy.

# 6. AI Companion Behavior

6.1. **Companion Task Queues**  
Each AI companion has a queue of tasks: follow, defend, gather. The scheduler prioritizes defend > gather > follow and can preempt current tasks.

6.2. **Emotional States**  
Companions maintain an “emotion meter” (0–100). Low morale (<30) reduces combat effectiveness; high morale (>80) grants buffs.

6.3. **Voice‐Directed Commands**  
The system hooks into a simple voice recognition module; spoken keywords map to task enqueueing.

# Conclusion

This document covers a broad array of mechanics—movement, combat, navigation, loot, events, and companion AI—that can be used to stress‑test your RAG pipeline’s chunking, indexing, and retrieval across multiple categories.  
