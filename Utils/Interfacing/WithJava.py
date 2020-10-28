from jnius import autoclass

System = autoclass('java.lang.System')
System.out.println('Hello World')

Stack = autoclass('java.util.Stack')
stack = Stack()
stack.push('a')
stack.push('b')

print(stack.pop())
print(stack.pop())

import jnius_config
import os
import subprocess

jnius_config.add_classpath('.')

java_mwe = """
public class MWE
{
    public void doSomething( long[] array ) {
        System.out.println("One argument.");
    }
    
    public void doSomething( long[] array, int[] otherArray ) {
        System.out.println("Two arguments.");
    }
    
}
"""

fp = 'MWE.java'
with open( fp, 'w' ) as f:
    f.write( java_mwe )

javac = '{}/bin/javac'.format(os.environ[ 'JAVA_HOME' ])
proc = subprocess.run( 
    [ javac, '-cp', jnius_config.split_char.join( jnius_config.get_classpath() ), fp ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
if proc.returncode != 0:
    print ( proc.stderr )

from jnius import autoclass

MWE = autoclass('MWE')
mwe = MWE()
a = [1,2,3]
b = [1,2,3]
mwe.doSomething(a)
mwe.doSomething(a, b)