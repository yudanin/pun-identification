#!/usr/bin/env python3
"""
Command-line interface for PIE - Pun Identification Engine

Usage:
    python cli.py "Time flies like an arrow; fruit flies like a banana."
    python cli.py --file sentences.txt
    python cli.py --interactive
"""

import argparse
import json
import os
import sys

from pie import PunIdentificationEngine


def print_result(result, verbose=False):
    """Pretty print analysis result."""
    print("\n" + "=" * 70)
    print(f"Sentence: {result.sentence}")
    print("=" * 70)
    
    if result.has_pun:
        print(f"\n✓ Found {len(result.puns)} pun(s):\n")
        
        for i, pun in enumerate(result.puns, 1):
            print(f"  [{i}] Word/Expression: \"{pun.word_or_expression}\"")
            print(f"      Type: {pun.pun_type.value}")
            print(f"      Sense 1: {pun.sense1}")
            print(f"      Sense 2: {pun.sense2}")
            
            if pun.frame_distance:
                print(f"      Frame Distance: {pun.frame_distance.distance:.1f} ({pun.frame_distance.distance_type})")
                print(f"      Frame Explanation: {pun.frame_distance.explanation}")
            
            print(f"      Explanation: {pun.explanation}")
            
            if pun.validation:
                print(f"      Confidence: {pun.confidence:.0%}")
                if verbose:
                    print(f"      Distributional Valid: {pun.validation.distributional_valid}")
                    print(f"      Substitution Valid: {pun.validation.substitution_valid}")
            print()
    else:
        print("\n✗ No puns detected.")
    
    if result.analysis_notes:
        print(f"Notes: {result.analysis_notes}")
    
    print()


def interactive_mode(engine):
    """Run in interactive mode."""
    print("\nPIE - Pun Identification Engine (Interactive Mode)")
    print("Enter sentences to analyze. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            sentence = input("Enter sentence: ").strip()
            
            if sentence.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if not sentence:
                continue
            
            result = engine.analyze(sentence)
            print_result(result, verbose=True)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PIE - Pun Identification Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'sentence',
        nargs='?',
        help='Sentence to analyze for puns'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='File containing sentences (one per line)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation tests'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("Error: No API key provided.")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    # Initialize engine
    try:
        engine = PunIdentificationEngine(
            api_key=api_key,
            validate=not args.no_validate
        )
    except Exception as e:
        print(f"Error initializing engine: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.interactive:
        interactive_mode(engine)
    
    elif args.file:
        # Process file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        results = engine.analyze_batch(sentences)
        
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for result in results:
                print_result(result, args.verbose)
    
    elif args.sentence:
        # Process single sentence
        result = engine.analyze(args.sentence)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result, args.verbose)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
