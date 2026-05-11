import json
import random
import os

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
NUM_SAMPLES = 50000  # Target 50k samples for high-quality reasoning
OUTPUT_FILE = "synthetic_nemotron_train.jsonl"

# --------------------------------------------------------------------------------
# Logic Rules & Generators
# --------------------------------------------------------------------------------

def generate_bit_manipulation():
    """Generates 8-bit manipulation puzzles (Mirror, Shift, XOR)"""
    bits = "".join(random.choice("01") for _ in range(8))
    rules = [
        ("Reverse the bits (Mirroring)", lambda b: b[::-1]),
        ("Shift left by 1 (Circular)", lambda b: b[1:] + b[0]),
        ("XOR with 10101010", lambda b: "".join(str(int(x) ^ int(y)) for x, y in zip(b, "10101010"))),
    ]
    rule_name, rule_fn = random.choice(rules)
    
    ex1_input = "".join(random.choice("01") for _ in range(8))
    ex1_output = rule_fn(ex1_input)
    
    prompt = f"In Alice's Wonderland, an 8-bit rule is applied. Example: {ex1_input} becomes {ex1_output}. What does {bits} become?"
    thinking = f"Thinking:\n1. Observe Example: {ex1_input} -> {ex1_output}.\n2. Identify Rule: {rule_name}.\n3. Apply to Input {bits}: {rule_fn(bits)}."
    answer = rule_fn(bits)
    
    return prompt, thinking, answer

def generate_symbol_digit():
    """Generates the high-difficulty Symbol-Digit puzzles (The 0.9 point bottleneck)"""
    # Mapping symbols to digits
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    mapping = {s: str(i % 10) for i, s in enumerate(symbols)}
    
    # 47 Common rules found in discussions (simplified for generation)
    rules = [
        ("Add digits and take modulo 10", lambda d1, d2: str((int(d1) + int(d2)) % 10)),
        ("Subtract and take absolute value", lambda d1, d2: str(abs(int(d1) - int(d2)))),
        ("Multiply and take last digit", lambda d1, d2: str((int(d1) * int(d2)) % 10)),
    ]
    rule_name, rule_fn = random.choice(rules)
    
    # Generate pair
    s1, s2 = random.sample(symbols, 2)
    s3, s4 = random.sample(symbols, 2)
    target_s1, target_s2 = random.sample(symbols, 2)
    
    ex1_in = s1 + s2
    ex1_out = rule_fn(mapping[s1], mapping[s2])
    ex2_in = s3 + s4
    ex2_out = rule_fn(mapping[s3], mapping[s4])
    
    prompt = f"Symbols transform to digits. Examples: {ex1_in} -> {ex1_out}, {ex2_in} -> {ex2_out}. Solve {target_s1}{target_s2}."
    
    thinking = (f"Thinking:\n1. Analyze Examples: {ex1_in}->{ex1_out}, {ex2_in}->{ex2_out}.\n"
                f"2. Infer Symbol values: {s1}={mapping[s1]}, {s2}={mapping[s2]}, {s3}={mapping[s3]}, {s4}={mapping[s4]}.\n"
                f"3. Identify Arithmetic Rule: {rule_name}.\n"
                f"4. Apply to {target_s1}{target_s2}: {mapping[target_s1]} and {mapping[target_s2]} -> {rule_fn(mapping[target_s1], mapping[target_s2])}.")
    
    answer = rule_fn(mapping[target_s1], mapping[target_s2])
    
    return prompt, thinking, answer

def generate_cipher():
    """Generates substitution cipher puzzles with reasoning"""
    vocab = ["APPLE", "BANANA", "CHERRY", "DRAGON", "ELDER", "FLOWER"]
    # Create random substitution
    abc = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    shuffled = abc[:]
    random.shuffle(shuffled)
    sub_map = dict(zip(abc, shuffled))
    
    def encrypt(word):
        return "".join(sub_map.get(c, c) for c in word)
    
    target_word = random.choice(vocab)
    ex_word = random.choice([w for w in vocab if w != target_word])
    
    prompt = f"In Alice's cipher, {ex_word} is written as {encrypt(ex_word)}. How is {target_word} written?"
    thinking = f"Thinking:\n1. Analyze substitution pattern from {ex_word} -> {encrypt(ex_word)}.\n2. Apply same mapping to characters in {target_word}.\n3. Result: {encrypt(target_word)}."
    answer = encrypt(target_word)
    
    return prompt, thinking, answer

# --------------------------------------------------------------------------------
# Main Generation Loop
# --------------------------------------------------------------------------------

def main():
    print(f"Generating {NUM_SAMPLES} samples to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(NUM_SAMPLES):
            # Mix the puzzle types
            choice = random.random()
            if choice < 0.4:  # 40% Symbol-Digit (The hardest type)
                p, t, a = generate_symbol_digit()
            elif choice < 0.7:  # 30% Bit Manipulation
                p, t, a = generate_bit_manipulation()
            else:  # 30% Cipher
                p, t, a = generate_cipher()
            
            # Format according to Nemotron instructions
            # Final answer must be in \boxed{}
            sample = {
                "instruction": p,
                "response": f"{t}\nThe final answer is \\boxed{{{a}}}."
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            if (i + 1) % 5000 == 0:
                print(f"Progress: {i + 1}/{NUM_SAMPLES} samples generated.")

    print("Generation complete!")

if __name__ == "__main__":
    main()
