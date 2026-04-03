const int dropletPin = 7;
const int speakerDrivePin = 5;
// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(dropletPin, OUTPUT);
  pinMode(speakerDrivePin, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);  // turn the LED on (HIGH is the voltage level)
  digitalWrite(dropletPin, LOW);
  digitalWrite(speakerDrivePin, LOW);
}
// the loop function runs over and over again forever
void loop() {
  digitalWrite(speakerDrivePin, HIGH);//activate speaker
  digitalWrite(LED_BUILTIN,HIGH);
  delay(2000);  
  digitalWrite(LED_BUILTIN,LOW);
  digitalWrite(dropletPin, HIGH);   //lift arm making droplet
  delay(9000);                      // wait for droplet to run its course
  digitalWrite(speakerDrivePin, LOW);   //remove signal from speaker
  digitalWrite(dropletPin, LOW);    //reset droplet arm ready for next droplet
  delay(1000);                      
}
