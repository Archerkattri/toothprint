# ToothPrint — terminal demo for VHS (charmbracelet/vhs).
# Install VHS + ttyd + ffmpeg (see video/README.md), then from the repo root:  vhs video/demo.tape
# Produces video/cli_demo.mp4 — a clean, reproducible terminal clip for the showcase.

Output video/cli_demo.mp4
Set Theme "Catppuccin Latte"
Set FontSize 26
Set Width 1280
Set Height 720
Set Padding 28
Set TypingSpeed 45ms

Type "# ToothPrint runs end-to-end with zero off-machine data:"
Enter
Sleep 1s
Type "TOOTHPRINT_FIXTURES=1 PYTHONPATH=. python evaluation/scripts/smoke_test.py"
Enter
Sleep 4s

Type "# Every result is reproduced by a committed script + JSON:"
Enter
Sleep 800ms
Type "cat evaluation/results/correspondence_identity.json | head -12"
Enter
Sleep 4s

Type "# Identity survives 50% tooth loss — 0.87 vs rigid GICP's 0.23."
Enter
Sleep 2.5s
