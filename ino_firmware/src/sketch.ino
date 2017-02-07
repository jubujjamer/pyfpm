// Import libraries
#include <SoftwareSerial.h>   // We need this even if we're not using a SoftwareSerial object
#include <SerialCommand.h>
#include <AccelStepper.h>
#include <Servo.h>
#include <Adafruit_NeoPixel.h>

// Defines for pinout
#define BAUD_RATE 9600
#define MOTOR_PINS 2 //Number of pins per motor
#define NOT_ENABLE_GENERAL 7 //Enables both motors
// #define STEPPER2_DIR_PIN 4 // CHANGED FROM 1!!
// #define STEPPER2_STEP_PIN 2
// #define STEPPER1_DIR_PIN 10 // CHECK
// #define STEPPER1_STEP_PIN 11 //CHECK
#define STEPPER2_DIR_PIN 4 // CHANGED FROM 1!!
#define STEPPER2_STEP_PIN 2
#define STEPPER1_DIR_PIN 9 // CHECK
#define STEPPER1_STEP_PIN 8 //CHECK
#define MS1 13
#define MS2 12
#define MS3 3
#define SERVO_CONTROL 5
#define SERVO_INIT_POS 0   // variable to store the servo position
#define SERVO_MAX_ANGLE 30
#define LC_PIN 6 // Laser power control pin
#define LEDS_DI 10
#define LED_NUMBER 14

// Initiate variables
SerialCommand sCmd; // Rename command
AccelStepper stepper(1, STEPPER2_STEP_PIN, STEPPER2_DIR_PIN);
Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

// Here I use Adafruit's library to control the NeoPixel Leds
Adafruit_NeoPixel strip = Adafruit_NeoPixel(LED_NUMBER, LEDS_DI, NEO_GRB+NEO_KHZ800);

void setup()
{
    pinMode(NOT_ENABLE_GENERAL,OUTPUT);
    //pinMode(LEDS_DI,OUTPUT);

    pinMode(LC_PIN,OUTPUT);
    pinMode(MS1,OUTPUT);
    pinMode(MS2,OUTPUT);
    pinMode(MS3,OUTPUT);
    digitalWrite(MS1,HIGH);
    digitalWrite(MS2,HIGH);
    digitalWrite(MS3,HIGH);
    digitalWrite(NOT_ENABLE_GENERAL,LOW);

  // Change these to suit your stepper if you want
    strip.begin();
    stepper.setMaxSpeed(189000);
    stepper.setAcceleration(20000);
    stepper.setCurrentPosition(0); // To move a predefined number of steps
    myservo.attach(SERVO_CONTROL);
    //// Setup callbacks for SerialCommand commands
    sCmd.addCommand("ROT", rotate); // Rotates the amount of steps stated in the argument
    sCmd.addCommand("LED",set_led_output); // sets the ws8
    //sCmd.setDefaultHandler(unrecognized); // Handler for command that isn't matched  (says "What?")
    // Start serial port communication with corresponding baud rate
    Serial.begin(BAUD_RATE);
    Serial.println("Ready");
}



void loop()
{
   //strip.show();
	while (Serial.available() > 0) {
		sCmd.readSerial();     // Process serial commands if new serial is available
		}
	stepper.run();
}

// Moving a user specified amount of positions function
void rotate() {
  // Validation
  char *arg;
  arg = sCmd.next();
  if (arg == NULL) {
    Serial.println("No steps or motor selected.");
    return;
  }
  int steps = atoi(arg); // Setting first argument as the desired amount of positions. Accepts negative steps
  //MoveEffector(Steps, Motor); // Caller for move executer function
  stepper.moveTo(int(steps));
  Serial.println(steps);
}

// Moving a user specified amount of positions function
void set_led_output() {
  // Validation
  char *arg;
  arg = sCmd.next();
  if (arg == NULL) {
    Serial.println("No something selected.");
    return;
  }
  int led = atoi(arg); //
  if (led < 0 || led > LED_NUMBER){
      Serial.println("Wrong led value.");
      return;
  }
  // setting red color
  arg = sCmd.next();
  if (arg == NULL) {
    Serial.println("No something selected.");
    return;
  }
  int mode = atoi(arg); //

  arg = sCmd.next();
  if (arg == NULL) {
    Serial.println("No something selected.");
    return;
  }
  int intensity = atoi(arg); //

  strip.clear();
  switch(mode){
    case 0:
      strip.setPixelColor(led, 255, 0, 0);
      break;
    case 1:
      strip.setPixelColor(led, 0, 255, 0);
      break;
    case 2:
      strip.setPixelColor(led, 0, 0, 255);
      break;
    case 3:
      strip.setPixelColor(led, 255, 255, 255);
      break;
  }
  strip.setBrightness(intensity);
  Serial.println(mode);
  strip.show();
}


// Private function that actually executes the steps
//void MoveEffector(int steps, int motor){
//  stepper.moveTo(steps);
  //Serial.println("HERE");
//}
// This gets set as the default handler, and gets called when no other command matches.
void unrecognized(const char *command) {
  Serial.println("Invalid Command");
}
