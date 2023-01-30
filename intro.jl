using DrWatson
@quickactivate "MAIS_Concept"
using Revise

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())
"""
)
