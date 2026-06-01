import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from control_msgs.msg import DynamicJointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import glfw
import threading
import sys
import tty
import termios
import select


class MinimalSubscriber(Node):

    def __init__(self):
            
        super().__init__('minimal_subscriber')


        custom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # 핵심: 되는 대로 다 받기
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            msg_type=DynamicJointState,
            topic='/dynamic_joint_states',
            callback=self.listener_callback,
            qos_profile=1
    
        )
        self.force = 0
        self.sensor_idx = 0
        self.save_count = 0
        self.sensor_names = ["finger_r_sensor1","finger_r_sensor2","finger_r_sensor3","finger_r_sensor4","finger_r_sensor5"
                        ,"finger_l_sensor1","finger_l_sensor2","finger_l_sensor3","finger_l_sensor4","finger_l_sensor5"]
        
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.data_buffer = None
        self.tac_data = []

        self.is_force_selected_flag = False
        self.save_flag = False
        self.save_file = False
        self.tac_data = []
    
    
    def listener_callback(self,msg):
        ## 보면서 저장하기
        ## 손가락 순서대로 받기
        
        if self.sensor_idx == 0:
            return
    
        sensor_name = self.sensor_names[self.sensor_idx-1]
        
        if sensor_name in msg.joint_names:
            idx = msg.joint_names.index(sensor_name)
            interface_data = msg.interface_values[idx]
            # self.get_logger().info(f"Collectd data: {interface_data.values}")

            if self.save_flag:
                self.save_flag = False
                
                if self.is_force_selected_flag == False:
                    self.get_logger().info(f"No force selected: please select force: f1 f2 f5 f10 f20")
                    return



                if self.tac_data is None:
                    self.tac_data = []

                self.save_count += 1
                self.data_buffer = interface_data.values
                self.tac_data.append(self.data_buffer)
                self.get_logger().info(f"Collectd data: {interface_data.values}")
                print(type(interface_data.values))
                self.get_logger().info(f"Counted: {self.save_count}")

            

            elif self.save_file:

                self.save_file = False

                if self.tac_data:

                    np.savez(f'tac_{sensor_name}_{self.force}N.npz',self.tac_data)
                    self.tac_data.clear()
                    self.is_force_selected_flag=False
                
                self.data_buffer = None
                self.get_logger().info(f"Collectd data: {sensor_name} saved {self.save_count} in force {self.force}N")
                self.save_count = 0


            # if self.force_changed_flag:
                


def input_thread(node):
    while rclpy.ok():

        user_input = input("입력 (force 값/ 센서번호/ a / s): ")

        if user_input.isdigit():
            val = int(user_input)
            if 0 <= val <= 10:
                node.sensor_idx = val
                if val > 0:
                    print(f"--- {node.sensor_names[val-1]} 관측 시작 ---")
                else:
                    print("--- 출력 중지 ---")

        elif user_input.startswith('f'):
            try:
                force_val = int(user_input[1:])
            except:
                print("잘못된 입력")
                continue
            force_val = int(user_input[1:])

            if 0<= force_val <= 20:
                node.force = force_val
                if force_val > 0:
                    print(f"--- {node.force}N 측정 시작 ---")
                else:
                    print("--- 출력 중지 ---")
            node.is_force_selected_flag = True
            


        elif user_input == 'a':
            node.save_flag = True

        elif user_input == 's':
            node.save_file = True
    


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    t = threading.Thread(target=input_thread, args=(minimal_subscriber,)) ## 여기서 node 공유됨. 
    t.daemon = True # 메인 프로그램 종료 시 함께 종료되도록 설정
    t.start()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


        

