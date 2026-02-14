#!/usr/bin/env python3
"""
Fix broken symlinks in organized dataset
Recreates symlinks with absolute paths
"""

from pathlib import Path


def fix_symlinks():
    """Fix broken symlinks in the organized dataset"""
    
    print("=" * 60)
    print("Fixing Symlinks")
    print("=" * 60)
    
    organized_dir = Path("data/raw/organized")
    audio_dir = Path("data/raw/nsynth-test/audio")
    
    if not organized_dir.exists():
        print(f"❌ Organized directory not found: {organized_dir}")
        return
    
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    total_fixed = 0
    total_errors = 0
    
    print(f"\n🔧 Fixing symlinks...")
    
    # Go through each instrument directory
    for class_dir in sorted(organized_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        fixed_count = 0
        
        # Check each symlink in this directory
        for symlink_file in class_dir.iterdir():
            if symlink_file.is_symlink():
                # Check if symlink is broken
                try:
                    symlink_file.resolve(strict=True)
                    # Symlink works, skip it
                    continue
                except FileNotFoundError:
                    # Symlink is broken, fix it
                    pass
                
                # Get the filename
                filename = symlink_file.stem
                src_file = audio_dir / f"{filename}.wav"
                
                if src_file.exists():
                    # Remove broken symlink
                    symlink_file.unlink()
                    
                    # Create new symlink with absolute path
                    try:
                        symlink_file.symlink_to(src_file.absolute())
                        fixed_count += 1
                        total_fixed += 1
                    except Exception as e:
                        print(f"   ❌ Failed to fix {symlink_file.name}: {e}")
                        total_errors += 1
                else:
                    print(f"   ❌ Source file not found: {src_file}")
                    total_errors += 1
        
        if fixed_count > 0:
            print(f"   ✓ {class_name:15s}: fixed {fixed_count} symlinks")
    
    print("\n" + "=" * 60)
    print(f"✅ Symlink Fix Complete!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   Fixed: {total_fixed} symlinks")
    if total_errors > 0:
        print(f"   Errors: {total_errors} symlinks")
    
    print(f"\n💡 Next: Test the audio loader again")
    print(f"   python src/preprocessing/audio_loader.py")
    print("=" * 60)


if __name__ == "__main__":
    fix_symlinks()
