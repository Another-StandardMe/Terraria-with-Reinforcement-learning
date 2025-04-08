import mmap
import struct
import time
import keyboard

MEMORY_MAP_NAME = "TerrariaRLState"
MEMORY_SIZE = 24  # 4字节 × 2（int） + 4字节 × 4（float） = 24 字节


class MemoryReader:
    def __init__(self, memory_map_name=MEMORY_MAP_NAME, memory_size=MEMORY_SIZE):
        self.memory_map_name = memory_map_name
        self.memory_size = memory_size

    def read_memory(self):
        """ 读取共享内存中的玩家数据 """
        try:
            with mmap.mmap(-1, self.memory_size, self.memory_map_name) as mm:
                mm.seek(0)
                data = mm.read(self.memory_size)
                if len(data) != self.memory_size:
                    raise ValueError(f"Unexpected data length: {len(data)} bytes")

                # 解析数据：两个 int 和四个 float
                return struct.unpack("ii4f", data)
        except Exception as e:
            print(f"Failed to read memory: {e}")
            return None


if __name__ == "__main__":
    reader = MemoryReader()
    while True:
        result = reader.read_memory()
        if result:
            player_hp, boss_hp, player_x, player_y, boss_x, boss_y = result
            print(f"Player HP: {player_hp}, Boss HP: {boss_hp}, Pos: ({player_x:.2f}, {player_y:.2f}), "
                  f"boss pos: ({boss_x:.2f}, {boss_y:.2f})")

        if keyboard.is_pressed("q"):
            print("\n'Q' pressed, exiting...")
            break

        time.sleep(0.02)
