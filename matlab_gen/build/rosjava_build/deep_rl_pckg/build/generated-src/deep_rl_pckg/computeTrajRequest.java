package deep_rl_pckg;

public interface computeTrajRequest extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "deep_rl_pckg/computeTrajRequest";
  static final java.lang.String _DEFINITION = "std_msgs/Float32MultiArray aqFunction\n";
  std_msgs.Float32MultiArray getAqFunction();
  void setAqFunction(std_msgs.Float32MultiArray value);
}
