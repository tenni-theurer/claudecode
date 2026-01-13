'use client';

export interface Task {
  id: string;
  title: string;
  completed: boolean;
}

interface TaskListProps {
  tasks: Task[];
}

export default function TaskList({ tasks }: TaskListProps) {
  if (tasks.length === 0) {
    return (
      <p className="text-gray-500 dark:text-gray-400 text-center py-8">
        No tasks yet. Add one above!
      </p>
    );
  }

  return (
    <ul className="w-full max-w-md space-y-2">
      {tasks.map((task) => (
        <li
          key={task.id}
          className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
        >
          <span className="flex-1 text-gray-800 dark:text-gray-200">
            {task.title}
          </span>
          {/* TODO: Add complete/delete buttons here */}
        </li>
      ))}
    </ul>
  );
}
