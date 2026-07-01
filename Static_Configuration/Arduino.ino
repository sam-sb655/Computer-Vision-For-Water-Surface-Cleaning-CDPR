#include <AccelStepper.h>

#define MOTOR_INTERFACE_TYPE 1

AccelStepper stepper1(MOTOR_INTERFACE_TYPE, 2, 3);
AccelStepper stepper2(MOTOR_INTERFACE_TYPE, 4, 5);
AccelStepper stepper3(MOTOR_INTERFACE_TYPE, 6, 7);
AccelStepper stepper4(MOTOR_INTERFACE_TYPE, 8, 9);

const byte BUF_SIZE = 64;
char inputBuf[BUF_SIZE];
byte bufIndex = 0;
bool commandReady = false;

void setup()
{
    Serial.begin(115200);

    stepper1.setMaxSpeed(1800);
    stepper2.setMaxSpeed(1800);
    stepper3.setMaxSpeed(1800);
    stepper4.setMaxSpeed(1800);

    stepper1.setAcceleration(700);
    stepper2.setAcceleration(700);
    stepper3.setAcceleration(700);
    stepper4.setAcceleration(700);

    Serial.println("Arduino Ready");
}

void readSerialNonBlocking()
{
    while (Serial.available() > 0)
    {
        char c = Serial.read();

        if (c == '\r') continue;

        if (c == '\n')
        {
            inputBuf[bufIndex] = '\0';
            bufIndex = 0;
            commandReady = true;
            return;
        }

        if (bufIndex < BUF_SIZE - 1)
        {
            inputBuf[bufIndex++] = c;
        }
    }
}

void processCommand()
{
    long s1, s2, s3, s4;
    int parsed = sscanf(inputBuf, "%ld,%ld,%ld,%ld", &s1, &s2, &s3, &s4);

    if (parsed == 4)
    {
        stepper1.move(-s1);
        stepper2.move(-s2);
        stepper3.move(-s3);
        stepper4.move(-s4);
    }
    else
    {
        Serial.println("ERR");
    }
}

void loop()
{
    readSerialNonBlocking();

    if (commandReady)
    {
        commandReady = false;
        processCommand();
    }

    stepper1.run();
    stepper2.run();
    stepper3.run();
    stepper4.run();
}
