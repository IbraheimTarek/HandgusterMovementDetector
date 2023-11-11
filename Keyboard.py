import keyboard

def on_arrow_key(event):
    if event.event_type == keyboard.KEY_DOWN:
        print(f"Arrow key pressed: {event.name}")

keyboard.hook_key('left', on_arrow_key)
keyboard.hook_key('right', on_arrow_key)
keyboard.hook_key('up', on_arrow_key)
keyboard.hook_key('down', on_arrow_key)

keyboard.wait('esc')
