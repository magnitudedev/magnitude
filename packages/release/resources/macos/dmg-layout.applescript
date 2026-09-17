on run argv
  set mountPath to item 1 of argv
  tell application id "com.apple.finder"
    set installerFolder to item (POSIX file mountPath as alias)
    open installerFolder
    tell container window of installerFolder
      set current view to icon view
      set toolbar visible to false
      set statusbar visible to false
      set bounds to {180, 140, 780, 542}
      set viewOptions to icon view options of container window of installerFolder
      set arrangement of viewOptions to not arranged
      set icon size of viewOptions to 88
      set text size of viewOptions to 13
      set background picture of viewOptions to file ".background:background.tiff" of installerFolder
      set position of item "Magnitude.app" of installerFolder to {145, 192}
      set position of item "Applications" of installerFolder to {455, 192}
      close
    end tell
    update installerFolder without registering applications
    delay 2
  end tell
end run
