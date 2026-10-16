#!/usr/bin/env python3
import rospy
import speech_recognition as sr
from std_msgs.msg import String
from drawbot_core import parse_voice_command


class VoiceNaturalCommandNode:
    def __init__(self):
        rospy.init_node("voice_natural_command_node", anonymous=False)

        self.command_topic = rospy.get_param("~command_topic", "/game_command")
        self.language = rospy.get_param("~language", "en-US")
        self.energy_threshold = rospy.get_param("~energy_threshold", 300)
        self.pause_threshold = rospy.get_param("~pause_threshold", 0.8)
        self.phrase_time_limit = rospy.get_param("~phrase_time_limit", 5)
        self.listen_timeout = rospy.get_param("~listen_timeout", 8)
        self.allow_cloud_stt = rospy.get_param("~allow_cloud_stt", False)

        self.pub = rospy.Publisher(self.command_topic, String, queue_size=10)

        self.recognizer = None
        self.microphone = None
        if self.allow_cloud_stt:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.dynamic_energy_threshold = True
            self.microphone = sr.Microphone()

        rospy.loginfo("Voice natural command node started.")
        rospy.loginfo("Publishing commands to: %s", self.command_topic)
        rospy.loginfo("Say: start the game / I finished drawing / please recognize it / stop")

        if self.allow_cloud_stt:
            rospy.logwarn(
                "Cloud speech recognition is enabled. Captured speech is sent to Google's service."
            )
            self.calibrate_microphone()
        else:
            rospy.logwarn(
                "Voice input is disabled by default because recognize_google sends audio to an "
                "external service. Set _allow_cloud_stt:=true only with informed consent."
            )

    def calibrate_microphone(self):
        rospy.loginfo("Calibrating microphone. Please keep quiet...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        rospy.loginfo("Microphone calibration finished.")
        rospy.loginfo("Energy threshold: %s", self.recognizer.energy_threshold)

    def text_to_command(self, text):
        rospy.loginfo("Recognized text: %s", text.lower().strip())
        return parse_voice_command(text)

    def run(self):
        if not self.allow_cloud_stt:
            rospy.logwarn("Voice node is idle; publish commands directly to %s.", self.command_topic)
            rospy.spin()
            return
        rospy.loginfo("Listening for voice commands...")

        while not rospy.is_shutdown():
            try:
                with self.microphone as source:
                    rospy.loginfo("Please speak now...")
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )

                rospy.loginfo("Recognizing speech...")
                text = self.recognizer.recognize_google(audio, language=self.language)

                command = self.text_to_command(text)

                if command:
                    rospy.loginfo("Mapped to game command: %s", command)
                    self.pub.publish(command)
                else:
                    rospy.logwarn("No command matched for: %s", text)

            except sr.WaitTimeoutError:
                rospy.logwarn("No speech detected. Try speaking louder or closer to the microphone.")
            except sr.UnknownValueError:
                rospy.logwarn("Speech was not understood.")
            except sr.RequestError as e:
                rospy.logerr("Speech recognition service error: %s", e)
                rospy.logerr("Google recognition needs internet. If this fails, use Vosk or Jupiter built-in voice.")
            except Exception as e:
                rospy.logerr("Voice node error: %s", e)


if __name__ == "__main__":
    try:
        node = VoiceNaturalCommandNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
