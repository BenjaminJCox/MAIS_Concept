using DrWatson
@quickactivate "MAIS_Concept"

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())
"""
)
