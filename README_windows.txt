AnyHershey

Requirements:
-Windows device
-Inkscape downloaded


Setup:
-Unzip anywhere
-Double-click AnyHershey.exe


Getting Started:
-Under 'Language preset', try selecting 'Latin (English, Spanish, etc.)' and typing anything into the text box. Then press the Generate SVG button to get a basic sense of how this tool works.
-Try switching 'Language preset' to 'Chinese (Simplified)' and typing anything, or copy this phrase '你好，世界' to see how the tool deals with logographic languages.


Preview:
-After you press Generate SVG, a preview window shows the lines exactly as they will be saved. Press Save to choose where to save the SVG, or Cancel to go back and change the settings.
-'White on black' switches the preview between white lines on black and black lines on white.
-The checkboxes on the right turn extra layers on and off:
	-mm grid: 1 mm lines (5 mm when zoomed far out), darker every 10 mm
	-Rasterized text (mask): the black-and-white image of the text that the lines were traced from
	-Skeleton pixels: that image thinned down to lines one pixel wide
	-Pen-up path: dashed moves between lines, with small arrows for direction
	-Strokes: the lines themselves
	-Length colors: strokes from orange (shortest, including dots) to blue (longest)
	-True pen width: strokes drawn as wide as the pen width you type in, to check where lines will run together
	-Junctions: where three or more lines of the skeleton meet
	-Stroke ends: a dot where each line starts and a ring where it ends
-The slider under the preview replays the plot in drawing order. Drag it, or press and hold < or > to play it backward or forward.
-'Export preview SVG' saves what the preview currently shows, with the layers you have on. This is separate from Save, which saves only the lines, for plotting.


Capabilities:
-Can produce a Hershey font for anything you can type and organizes strokes for cutting/plotting
-Has fine-tuned settings for a number of common written languages:
	-English
	-Chinese (Simplified)
	-Chinese (Traditional)
	-Kanji (Japanese)
	-Hiragana (Japanese)
	-Katakana (Japanese)
	-Hangul (Korean)
	-Devanagari (Hindi-family)
	-Arabic


Notes:
-This project runs Inkscape in the background. Occasionally, the firewall can block this and you will need to let it through explicitly
-Whichever font you select must be installed on your computer. If it is not, a warning appears under 'Font family' and Inkscape uses its default font instead (Verdana on most computers). See how to add fonts here: https://support.microsoft.com/en-us/windows/manage-fonts-in-windows-f12d0657-2fc8-7613-c76f-88d043b334b8
-If converting a chunk of text that contains multiple writing systems, start with the 'Default' language preset and then fine tune from there.
