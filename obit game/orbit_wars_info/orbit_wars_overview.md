# Orbit Wars | Kaggle Competition Overview

## 🌌 Introduction
Orbit Wars is a real-time strategy game set in continuous 2D space where the objective is to conquer planets rotating around a central sun. The competition requires participants to create and train AI bots to compete in 1v1 or 4-player Free-For-All (FFA) matches against other submitted agents.

---

## 🎮 Gameplay Mechanics
* **Space/Board**: The game takes place on a 100x100 continuous board with a sun located at the center (radius 10). Colliding with the sun destroys fleets.
* **Planets**: Planets can be static or orbiting. When owned, they generate new ships every turn (between 1 and 5). 
* **Fleets**: Players can launch fleets from their planets to capture others. Fleets move in straight lines, and their movement speed scales with the size of the fleet.
* **Comets**: These are temporary celestial objects that fly through the board on elliptical orbits, providing temporary production boosts if captured.
* **Combat System**: Battles are resolved when opposing fleets collide with planets. The largest force survives with the difference in ship counts.

---

## 🏆 Evaluation & Ranking
* Agents are evaluated and ranked using a **Gaussian skill rating** model: `N(μ, σ²)`.
* Before entering the main matchmaking pool, agents must pass validation episodes to ensure they function correctly.
* Your top 2 submissions will be used to determine your final ranking on the leaderboard.

---

## 📅 Important Dates (Timeline)
* **Start Date**: April 16, 2026
* **Entry / Team Merger Deadline**: June 16, 2026
* **Final Submission Deadline**: June 23, 2026
* **Leaderboard Finalization**: ~July 8, 2026

*(Note: All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted.)*

---

## 💰 Prizes
A total prize pool of **$50,000** is available.
* **1st to 10th Place**: $5,000 each.

---

## 📜 Key Rules
* **Team Limits**: Maximum of 5 participants per team.
* **Daily Submissions**: Maximum of 5 submissions per day.
* **External Data**: Allowed, provided it is publicly available and reasonably accessible to all participants.
* **Winner Obligations**: Winning solutions must be open-sourced under the **CC-BY 4.0 license**.
* **Prohibitions**: Private sharing of code outside of teams is strictly prohibited. Multiple accounts per user are not allowed.

---

## 📂 Data & Starter Kit
Participants are provided with a Python starter kit to help jumpstart development. 
* The kit includes: `README.md`, `agents.md`, and `main.py`.
* The competition environment is built using Kaggle Environments (Simulation).

---
> Source: [Kaggle: Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/overview/description)
