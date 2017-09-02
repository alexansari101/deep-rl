package deep_rl_pckg;

public interface computeTrajResponse extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "deep_rl_pckg/computeTrajResponse";
  static final java.lang.String _DEFINITION = "std_msgs/Float32MultiArray poseAgent";
  std_msgs.Float32MultiArray getPoseAgent();
  void setPoseAgent(std_msgs.Float32MultiArray value);
}
