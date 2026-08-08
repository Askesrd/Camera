import serial
import time
ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)
print(ser.readline().decode().strip())

while True:
    a=input()
    if a.lower() == 'q':
        ser.close()
        break
    cmd = f"{a}\n"
    ser.write(cmd.encode())
    while True:
        q = ser.readline().decode().strip()
        if not q:
            continue
        else:
            print(q)
            break


